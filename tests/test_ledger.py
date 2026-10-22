import json

from atl.ledger.chain import Chain


def test_chain_verifies_and_detects_tampering(tmp_path):
    c = Chain(tmp_path / "l.jsonl")
    for i in range(20):
        c.append("MARK", {"book": "live", "equity": 100000 + i})
    ok, msg = c.verify()
    assert ok, msg
    lines = (tmp_path / "l.jsonl").read_text().splitlines()
    rec = json.loads(lines[7])
    rec["data"]["equity"] = 999999          # rewrite history
    lines[7] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    (tmp_path / "l.jsonl").write_text("\n".join(lines) + "\n")
    ok, msg = Chain(tmp_path / "l.jsonl").verify()
    assert not ok and "seq 7" in msg


def test_chain_detects_deletion(tmp_path):
    c = Chain(tmp_path / "l.jsonl")
    for i in range(5):
        c.append("X", {"i": i})
    lines = (tmp_path / "l.jsonl").read_text().splitlines()
    del lines[2]
    (tmp_path / "l.jsonl").write_text("\n".join(lines) + "\n")
    ok, _ = Chain(tmp_path / "l.jsonl").verify()
    assert not ok


def test_append_continues_chain_across_instances(tmp_path):
    Chain(tmp_path / "l.jsonl").append("A", {"x": 1})
    Chain(tmp_path / "l.jsonl").append("B", {"x": 2})
    ok, msg = Chain(tmp_path / "l.jsonl").verify()
    assert ok and "2 records" in msg


def test_interleaved_writers_keep_the_chain_intact(tmp_path):
    a, b = Chain(tmp_path / "l.jsonl"), Chain(tmp_path / "l.jsonl")
    for i in range(6):
        (a if i % 2 else b).append("X", {"i": i})
    ok, msg = Chain(tmp_path / "l.jsonl").verify()
    assert ok, msg
