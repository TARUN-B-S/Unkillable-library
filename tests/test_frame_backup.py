import os
from pathlib import Path

from unkillable.backup import frame_codec
from unkillable.backup.daily_backup import backup_daily
from unkillable.backup.frame_restore import restore_frames


def test_frame_roundtrip_small():
    data = os.urandom(50_000)
    frames = frame_codec.encode_bytes(data)
    assert frames
    for img, _ in frames:
        assert img.size == (1920, 1080) and img.mode == "L"
    back = frame_codec.decode_images([img for img, _ in frames])
    assert back == data


def test_frame_corrects_small_corruption(tmp_path):
    data = os.urandom(20_000)
    frames = frame_codec.encode_bytes(data)
    img, _ = frames[0]
    arr_path = tmp_path / "f.png"
    img.save(arr_path)
    import numpy as np
    from PIL import Image

    a = np.array(Image.open(arr_path))
    rng = np.random.default_rng(0)
    ys = rng.integers(100, 900, size=200)
    xs = rng.integers(100, 1800, size=200)
    for y, x in zip(ys, xs):
        a[y, x] = (int(a[y, x]) + 17) % 256
    noisy = Image.fromarray(a, mode="L")
    back = frame_codec.decode_images([noisy])
    assert back == data


def test_frame_detects_heavy_corruption():
    import numpy as np

    data = os.urandom(20_000)
    frames = frame_codec.encode_bytes(data)
    img, _ = frames[0]
    a = np.array(img)
    a[40:340, 40:1540] = 0
    from PIL import Image

    bad = Image.fromarray(a, mode="L")
    try:
        back = frame_codec.decode_images([bad])
    except Exception:
        return
    assert back != data


def test_daily_backup_and_restore(tmp_path):
    src = tmp_path / "storage"
    (src / "clips").mkdir(parents=True)
    (src / "clips" / "a.txt").write_text("hello frames")
    (src / "events.jsonl").write_text('{"camera":"t"}\n')
    dest = tmp_path / "backups"
    res = backup_daily(src, dest, date_str="2026-09-12", verify=True, delete_source=False)
    assert Path(res["dir"]).exists()
    out = tmp_path / "restored"
    r = restore_frames(res["dir"], out)
    assert r["bytes"] > 0
    assert (out / "storage" / "clips" / "a.txt").read_text() == "hello frames"


def test_delete_guard(tmp_path):
    import pytest

    from unkillable.backup.daily_backup import BackupError

    src = tmp_path / "s"
    src.mkdir()
    (src / "x.txt").write_text("x")
    with pytest.raises(BackupError):
        backup_daily(src, tmp_path / "b", date_str="2026-09-12", verify=False, delete_source=True)


def test_parity_frame_group_roundtrip(monkeypatch):
    monkeypatch.setattr(frame_codec, "DATA_PER_FRAME", 5000)
    monkeypatch.setattr(frame_codec, "PARITY_GROUP", 4)
    data = os.urandom(22_000)
    frames = frame_codec.encode_bytes(data)
    assert any(m["flags"] == 1 for _, m in frames)
    back = frame_codec.decode_images([img for img, _ in frames], total_bytes=len(data))
    assert back == data


def test_parity_recovers_lost_data_frame(monkeypatch):
    monkeypatch.setattr(frame_codec, "DATA_PER_FRAME", 5000)
    monkeypatch.setattr(frame_codec, "PARITY_GROUP", 4)
    data = os.urandom(22_000)
    frames = frame_codec.encode_bytes(data)
    imgs = [img for img, m in frames if m["flags"] == 0]
    kept = [img for i, img in enumerate(imgs) if i != 1]
    assert len(kept) == len(imgs) - 1
    parity = [img for img, m in frames if m["flags"] == 1]
    back = frame_codec.decode_images(kept + parity, total_bytes=len(data))
    assert back == data
