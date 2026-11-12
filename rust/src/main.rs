use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::json;
use shakmaty::{
    Bitboard, CastlingMode, Chess, Color, Position, Square, attacks,
    fen::{Epd, Fen},
    uci::UciMove,
};
use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    io::{self, BufRead, BufReader, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use zip::{ZipWriter, write::SimpleFileOptions};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: String,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 0)]
    limit: usize,
    #[arg(long, default_value_t = 10000)]
    shard_size: usize,
    #[arg(long, default_value_t = 20000)]
    batch_size: usize,
    #[arg(long, default_value_t = 8)]
    threads: usize,
    #[arg(long, default_value = "unversioned")]
    revision: String,
    #[arg(long, default_value = "moves.json")]
    vocab: PathBuf,
    /// Omit masks (the default). Generate legal masks at training time.
    #[arg(long, conflicts_with = "masks")]
    no_masks: bool,
    /// Include Python-compatible packed legal masks.
    #[arg(long)]
    masks: bool,
}
#[derive(Deserialize)]
struct Record {
    fen: String,
    evals: Vec<Eval>,
}
#[derive(Deserialize)]
struct Eval {
    depth: i64,
    #[serde(default)]
    knodes: i64,
    pvs: Vec<Pv>,
}
#[derive(Deserialize)]
struct Pv {
    line: String,
    cp: Option<serde_json::Number>,
    mate: Option<i64>,
}
struct Row {
    tokens: [u8; 128],
    label: u16,
    legal: Option<[u8; 246]>,
    key: u64,
    cp: f32,
    mate: i16,
    split: u8,
}

fn canonical_uci(m: shakmaty::Move, black: bool) -> String {
    let mut u = UciMove::from_standard(m);
    if black {
        if let UciMove::Normal {
            from,
            to,
            promotion,
        } = u
        {
            u = UciMove::Normal {
                from: from.flip_vertical(),
                to: to.flip_vertical(),
                promotion,
            };
        }
    }
    u.to_string()
}
fn encode(line: &str, ids: &HashMap<String, u16>, masks: bool) -> Result<Row> {
    let r: Record = serde_json::from_str(line)?;
    // max_by_key uses the LAST tie; use strictly greater updates instead.
    let mut best = r.evals.first().context("empty evals")?;
    for ev in &r.evals[1..] {
        if (ev.depth, ev.knodes) > (best.depth, best.knodes) {
            best = ev;
        }
    }
    let pv = best.pvs.first().context("empty pvs")?;
    let fen: Fen = r
        .fen
        .parse()
        .map_err(|_| anyhow!("invalid orthodox board"))?;
    let setup = fen.into_setup();
    for color in Color::ALL {
        if setup.board.by_color(color).count() > 16
            || (setup.board.by_color(color) & setup.board.pawns()).count() > 8
        {
            bail!("invalid orthodox board");
        }
    }
    // shakmaty rejects excess promoted material and some double-stepper checks
    // that python-chess accepts. Apply Python's weaker validity checks below.
    let pos: Chess = setup
        .clone()
        .position(CastlingMode::Standard)
        .or_else(|e| e.ignore_too_much_material())
        .or_else(|e| e.ignore_impossible_check())
        .map_err(|_| anyhow!("invalid orthodox board"))?;
    let checkers = pos.checkers();
    if checkers.count() > 2 {
        bail!("invalid orthodox board");
    }
    if let Some(king) = pos.board().king_of(pos.turn()) {
        if !checkers.is_empty() {
            if let Some(ep) = setup.ep_square {
                let pushed_to = Square::new((u32::from(ep)) ^ 8);
                let pushed_from = Square::new((u32::from(ep)) ^ 24);
                let occupied = pos.board().occupied().without(pushed_to).with(pushed_from);
                if checkers.count() > 1
                    || (checkers.first() != Some(pushed_to)
                        && !pos.king_attackers(king, !pos.turn(), occupied).is_empty())
                {
                    bail!("invalid orthodox board");
                }
            } else if checkers.count() == 2
                && attacks::aligned(checkers.first().unwrap(), king, checkers.last().unwrap())
            {
                bail!("invalid orthodox board");
            }
        }
    }
    let uci: UciMove = pv
        .line
        .split_whitespace()
        .next()
        .context("empty line")?
        .parse()?;
    let m = uci.to_move(&pos).map_err(|_| anyhow!("illegal target"))?;
    let black = pos.turn() == Color::Black;
    let label = *ids
        .get(&canonical_uci(m, black))
        .context("target not in vocabulary")?;
    let mut s = setup;
    if black {
        s.mirror();
    }
    let mut tokens = [0u8; 128];
    for (i, token) in tokens[..64].iter_mut().enumerate() {
        *token = s.board.piece_at(Square::new(i as u32)).map_or(1, |p| {
            1 + p.role as u8 + if p.color == Color::White { 0 } else { 6 }
        });
    }
    for (i, sq) in [Square::H1, Square::A1, Square::H8, Square::A8]
        .iter()
        .enumerate()
    {
        tokens[64 + i] = 14 + u8::from(s.castling_rights.contains(*sq));
    }
    tokens[68] = s.ep_square.map_or(16, |sq| 17 + sq.file() as u8);
    // EPD preserves the literal FEN ep square even if no capture is possible.
    // python-chess drops promoted markers for orthodox FEN serialization.
    s.promoted = Bitboard::EMPTY;
    s.pockets = None;
    s.remaining_checks = None;
    let key_fen = Epd::try_from_setup(s)
        .map_err(|_| anyhow!("bad canonical FEN"))?
        .to_string();
    let hash = blake2b_simd::Params::new()
        .hash_length(8)
        .hash(key_fen.as_bytes());
    let key = u64::from_le_bytes(hash.as_bytes().try_into().unwrap());
    let legal = if masks {
        let mut mask = [0u8; 246];
        for m in pos.legal_moves() {
            let id = *ids
                .get(&canonical_uci(m, black))
                .context("legal move absent")? as usize;
            mask[id / 8] |= 1 << (id % 8);
        }
        Some(mask)
    } else {
        None
    };
    let sign = if black { -1.0 } else { 1.0 };
    let mate = pv.mate.unwrap_or(0) * if black { -1 } else { 1 };
    if mate < i16::MIN as i64 || mate > i16::MAX as i64 {
        bail!("mate out of range");
    }
    Ok(Row {
        tokens,
        label,
        legal,
        key,
        cp: pv.cp.as_ref().map_or(f32::NAN, |x| {
            if x.is_i64() {
                (x.as_i64().unwrap() as f64 * sign + 0.0) as f32
            } else {
                (x.as_f64().unwrap() * sign) as f32
            }
        }),
        mate: mate as i16,
        split: if key % 1000 < 980 {
            0
        } else if key % 1000 < 990 {
            1
        } else {
            2
        },
    })
}
fn npy(
    zip: &mut ZipWriter<File>,
    name: &str,
    dtype: &str,
    shape: &[usize],
    bytes: &[u8],
) -> Result<()> {
    // NPY v1.0, 64-byte aligned header as in NumPy; little-endian scalars.
    let shape = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        dtype, shape
    );
    let padding = (64 - ((10 + header.len() + 1) % 64)) % 64;
    header.push_str(&" ".repeat(padding));
    header.push('\n');
    zip.start_file(
        format!("{name}.npy"),
        SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored),
    )?;
    zip.write_all(b"\x93NUMPY\x01\x00")?;
    zip.write_all(&(header.len() as u16).to_le_bytes())?;
    zip.write_all(header.as_bytes())?;
    zip.write_all(bytes)?;
    Ok(())
}
fn flush(out: &Path, shard: usize, rows: &mut Vec<Row>, masks: bool) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let path = out.join(format!("shard-{shard:06}.npz"));
    let tmp = path.with_extension("npz.partial");
    let mut z = ZipWriter::new(File::create(&tmp)?);
    let n = rows.len();
    npy(
        &mut z,
        "tokens",
        "|u1",
        &[n, 128],
        &rows.iter().flat_map(|r| r.tokens).collect::<Vec<_>>(),
    )?;
    npy(
        &mut z,
        "labels",
        "<u2",
        &[n],
        &rows
            .iter()
            .flat_map(|r| r.label.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    if masks {
        npy(
            &mut z,
            "legal",
            "|u1",
            &[n, 246],
            &rows
                .iter()
                .flat_map(|r| r.legal.unwrap())
                .collect::<Vec<_>>(),
        )?;
    }
    npy(
        &mut z,
        "keys",
        "<u8",
        &[n],
        &rows
            .iter()
            .flat_map(|r| r.key.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    npy(
        &mut z,
        "cp",
        "<f4",
        &[n],
        &rows
            .iter()
            .flat_map(|r| r.cp.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    npy(
        &mut z,
        "mate",
        "<i2",
        &[n],
        &rows
            .iter()
            .flat_map(|r| r.mate.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    npy(
        &mut z,
        "split",
        "|u1",
        &[n],
        &rows.iter().map(|r| r.split).collect::<Vec<_>>(),
    )?;
    z.finish()?.sync_all()?;
    fs::rename(tmp, path)?;
    rows.clear();
    Ok(())
}
fn main() -> Result<()> {
    let a = Args::parse();
    if a.shard_size == 0 || a.batch_size == 0 || a.threads == 0 {
        bail!("sizes and threads must be positive");
    }
    let vocab: Vec<String> = serde_json::from_reader(File::open(&a.vocab)?)?;
    if vocab.len() != 1968 {
        bail!("expected 1968 vocabulary entries");
    }
    let ids: HashMap<_, _> = vocab
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i as u16))
        .collect();
    if ids.len() != 1968 {
        bail!("duplicate vocabulary");
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(a.threads)
        .build_global()?;
    fs::create_dir(&a.output).context("output must not already exist")?;
    let raw: Box<dyn io::Read> = if a.input == "-" {
        Box::new(io::stdin())
    } else {
        Box::new(File::open(&a.input)?)
    };
    let input: Box<dyn io::Read> = if a.input.ends_with(".zst") {
        Box::new(zstd::stream::read::Decoder::new(raw)?)
    } else {
        raw
    };
    let mut reader = BufReader::with_capacity(4 * 1024 * 1024, input);
    let (mut rows, mut accepted, mut shards) = (0usize, 0usize, 0usize);
    let mut splits = [0usize; 3];
    let mut errors = BTreeMap::<String, usize>::new();
    let mut buf = Vec::with_capacity(a.shard_size);
    let start = Instant::now();
    loop {
        let mut batch = Vec::with_capacity(a.batch_size);
        for _ in 0..a.batch_size {
            if a.limit != 0 && rows + batch.len() >= a.limit {
                break;
            }
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            batch.push(line);
        }
        if batch.is_empty() {
            break;
        }
        rows += batch.len();
        // Indexed parallel collection preserves the original input order.
        let results: Vec<_> = batch.par_iter().map(|l| encode(l, &ids, a.masks)).collect();
        for result in results {
            match result {
                Ok(r) => {
                    accepted += 1;
                    splits[r.split as usize] += 1;
                    buf.push(r);
                    if buf.len() == a.shard_size {
                        flush(&a.output, shards, &mut buf, a.masks)?;
                        shards += 1;
                    }
                }
                Err(e) => {
                    *errors.entry(e.to_string()).or_default() += 1;
                }
            }
        }
        eprintln!(
            "rows={rows} accepted={accepted} rows/s={:.0}",
            rows as f64 / start.elapsed().as_secs_f64()
        );
    }
    if !buf.is_empty() {
        flush(&a.output, shards, &mut buf, a.masks)?;
        shards += 1;
    }
    let seconds = start.elapsed().as_secs_f64();
    let manifest = json!({"rows":rows,"accepted":accepted,"rejected":rows-accepted,
        "train":splits[0],"val":splits[1],"test":splits[2],"errors":errors,
        "seconds":seconds,"rows_per_second":rows as f64/seconds,"shards":shards,
        "source":a.input,"source_revision":a.revision,"selection":"max(depth,knodes,earliest eval); pvs[0]",
        "score_pov":"source White; output side to move","token_length":128,"active_tokens":69,
        "vocab_size":1968,"split":"canonical FEN blake2b64 modulo1000: 980/10/10",
        "schema_version":1,"legal_masks":a.masks,"threads":a.threads});
    fs::write(a.output.join("moves.json"), serde_json::to_vec(&vocab)?)?;
    fs::write(
        a.output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    println!("{manifest}");
    Ok(())
}
