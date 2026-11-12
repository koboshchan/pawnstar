//! Ordered streaming over HF's flattened PV rows. Carry the selected row across files.
use anyhow::{Context, Result, bail};
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::{Field, Row},
};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::mpsc::{SyncSender, sync_channel},
};

pub fn files(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_owned()]);
    }
    let dir = if input.join("data").is_dir() {
        input.join("data")
    } else {
        input.to_owned()
    };
    let mut paths = fs::read_dir(dir)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.retain(|p| p.extension().is_some_and(|e| e == "parquet"));
    paths.sort();
    if paths.is_empty() {
        bail!("no parquet shards found");
    }
    Ok(paths)
}
fn field(row: &Row, name: &str) -> Result<Value> {
    let f = row
        .get_column_iter()
        .find(|(n, _)| n.as_str() == name)
        .with_context(|| format!("missing HF column {name}"))?
        .1;
    Ok(match f {
        Field::Null => Value::Null,
        Field::Str(s) => json!(s),
        Field::Byte(v) => json!(v),
        Field::Short(v) => json!(v),
        Field::Int(v) => json!(v),
        Field::Long(v) => json!(v),
        Field::UByte(v) => json!(v),
        Field::UShort(v) => json!(v),
        Field::UInt(v) => json!(v),
        Field::ULong(v) => json!(v),
        _ => bail!("unexpected type for HF column {name}"),
    })
}
struct Selected {
    fen: String,
    depth: i64,
    knodes: i64,
    pv: Value,
}
impl Selected {
    fn parse(row: Row) -> Result<Self> {
        let fen = field(&row, "fen")?
            .as_str()
            .context("fen must be string")?
            .to_owned();
        let line = field(&row, "line")?;
        if !line.is_string() {
            bail!("line must be string");
        }
        Ok(Self {
            fen,
            depth: field(&row, "depth")?
                .as_i64()
                .context("depth must be integer")?,
            knodes: field(&row, "knodes")?
                .as_i64()
                .context("knodes must be integer")?,
            pv: json!({"line":line,"cp":field(&row,"cp")?,"mate":field(&row,"mate")?}),
        })
    }
    fn json(self) -> String {
        json!({"fen":self.fen,"evals":[{"depth":self.depth,"knodes":self.knodes,"pvs":[self.pv]}]})
            .to_string()
    }
}
fn produce(paths: Vec<PathBuf>, tx: &SyncSender<Result<String>>) -> Result<()> {
    let mut selected: Option<Selected> = None;
    for path in paths {
        let reader = SerializedFileReader::new(File::open(&path)?)?;
        for row in reader.get_row_iter(None)? {
            let next = Selected::parse(row?)?;
            match selected.as_mut() {
                Some(best) if best.fen == next.fen => {
                    // STRICT greater keeps the first PV of the earliest tied evaluation.
                    if (next.depth, next.knodes) > (best.depth, best.knodes) {
                        *best = next;
                    }
                }
                _ => {
                    if let Some(done) = selected.take()
                        && tx.send(Ok(done.json())).is_err()
                    {
                        return Ok(());
                    }
                    selected = Some(next);
                }
            }
        }
    }
    if let Some(done) = selected {
        let _ = tx.send(Ok(done.json()));
    }
    Ok(())
}
pub fn records(paths: Vec<PathBuf>) -> impl Iterator<Item = Result<String>> {
    let (tx, rx) = sync_channel(256);
    std::thread::spawn(move || {
        if let Err(e) = produce(paths, &tx) {
            let _ = tx.send(Err(e));
        }
    });
    rx.into_iter()
}
