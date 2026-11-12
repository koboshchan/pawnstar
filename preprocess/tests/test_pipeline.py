import json,subprocess,sys
import chess
def test_pipeline(tmp_path):
    source=tmp_path/"source.jsonl"
    source.write_text(json.dumps({"fen":chess.STARTING_FEN,"evals":[{"depth":20,"pvs":[{"cp":50,"line":"d2d4"}]},{"depth":30,"pvs":[{"cp":19,"line":"e2e4"},{"cp":50,"line":"d2d4"}]}]})+"\n")
    out=tmp_path/"output"
    subprocess.run([sys.executable,"preprocess.py","--input",str(source),"--output",str(out)],check=True)
    subprocess.run([sys.executable,"verify.py",str(out)],check=True)
    m=json.loads((out/"manifest.json").read_text())
    assert m["rows"]==m["accepted"]==1
