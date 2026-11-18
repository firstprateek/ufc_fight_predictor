import shutil
from pathlib import Path

import httpx

from ufc import fetch as F
from ufc.cli import main

FIX = Path(__file__).parent / "fixtures" / "source"
SHA = "a" * 40


def handler(request: httpx.Request) -> httpx.Response:
    if request.url.host == "api.github.com":
        return httpx.Response(200, text=SHA + "\n")
    name = request.url.path.rsplit("/", 1)[-1]
    assert request.url.path.startswith(f"/{F.REPO}/{SHA}/")
    return httpx.Response(200, content=(FIX / name).read_bytes())


def test_fetch_resolves_sha_and_caches(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    snap = F.fetch(tmp_path, client=client)
    assert snap == tmp_path / SHA and F.latest_snapshot(tmp_path) == snap
    assert sorted(p.stem for p in snap.glob("*.csv")) == sorted(F.FILES)
    # second run: nothing is downloaded again
    calls = []
    client2 = httpx.Client(transport=httpx.MockTransport(lambda r: calls.append(r) or handler(r)))
    F.fetch(tmp_path, sha=SHA, client=client2)
    assert calls == []


def test_cli_build_and_validate(tmp_path, capsys):
    snap = tmp_path / "raw" / SHA
    shutil.copytree(FIX, snap)
    (tmp_path / "raw" / "latest.txt").write_text(SHA)
    ov = FIX.parent / "overrides"
    assert main(["--data-dir", str(tmp_path), "--overrides", str(ov), "build"]) == 0
    out = capsys.readouterr().out
    assert "fights: 8 rows" in out and "validation passed" in out
    assert main(["--data-dir", str(tmp_path), "validate"]) == 0


def test_cli_build_without_fetch_fails(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path), "build"]) == 1
    assert "ufc fetch" in capsys.readouterr().err
