import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)

WIDTH = 1920
HEIGHT = 1080
BORDER = 40
CORNER = 24
MAGIC = b"UKLB1"
VERSION = 1
HEADER_LEN = 64
HEADER_REPEAT = 3
HEADER_AREA = HEADER_LEN * HEADER_REPEAT
INNER_W = WIDTH - 2 * BORDER
INNER_H = HEIGHT - 2 * BORDER
INNER_CAP = INNER_W * INNER_H
DATA_BLOCK = 239
CODE_BLOCK = 255
BLOCKS_PER_FRAME = (INNER_CAP - HEADER_AREA - 4 * CORNER * CORNER) // CODE_BLOCK
DATA_PER_FRAME = BLOCKS_PER_FRAME * DATA_BLOCK
PARITY_GROUP = 10

_HEADER_STRUCT = struct.Struct(">5sBIIIII38x")


class FrameCodecError(RuntimeError):
    pass


def _get_codec(nsym: int = 16):
    from reedsolo import RSCodec

    return RSCodec(nsym)


def _pack_header(total: int, idx: int, payload_len: int, data_crc: int, flags: int = 0) -> bytes:
    return _HEADER_STRUCT.pack(MAGIC, VERSION, total, idx, payload_len, data_crc, flags)


def _unpack_header(raw: bytes) -> dict:
    if len(raw) < HEADER_LEN:
        raise FrameCodecError("Header too short")
    try:
        magic, ver, total, idx, payload_len, data_crc, flags = _HEADER_STRUCT.unpack(raw[:HEADER_LEN])
    except struct.error as exc:
        raise FrameCodecError(f"Bad header: {exc}") from exc
    if magic != MAGIC:
        raise FrameCodecError(f"Bad magic: {magic!r}")
    if ver != VERSION:
        raise FrameCodecError(f"Unsupported version: {ver}")
    return {"total": total, "idx": idx, "payload_len": payload_len, "data_crc": data_crc, "flags": flags}


def _vote_header(blob: bytes) -> bytes:
    cands = [blob[i * HEADER_LEN:(i + 1) * HEADER_LEN] for i in range(HEADER_REPEAT)]
    cands = [c for c in cands if len(c) == HEADER_LEN and c[:5] == MAGIC]
    if not cands:
        raise FrameCodecError("No valid header copy found")
    for c in cands:
        if cands.count(c) >= 2:
            return c
    return cands[0]


def _corner_mask() -> np.ndarray:
    m = np.ones((INNER_H, INNER_W), dtype=bool)
    m[:CORNER, :CORNER] = False
    m[:CORNER, -CORNER:] = False
    m[-CORNER:, :CORNER] = False
    m[-CORNER:, -CORNER:] = False
    return m


def _rs_encode_chunk(chunk: bytes, nsym: int = 16) -> bytes:
    codec = _get_codec(nsym)
    out = bytearray()
    for i in range(0, len(chunk), DATA_BLOCK):
        block = chunk[i:i + DATA_BLOCK]
        if len(block) < DATA_BLOCK:
            block = block + b"\x00" * (DATA_BLOCK - len(block))
        out += codec.encode(block)
    return bytes(out)


def _rs_decode_chunk(coded: bytes, payload_len: int, nsym: int = 16) -> bytes:
    codec = _get_codec(nsym)
    nblocks = len(coded) // CODE_BLOCK
    out = bytearray()
    for i in range(nblocks):
        block = coded[i * CODE_BLOCK:(i + 1) * CODE_BLOCK]
        try:
            dec = codec.decode(block)
            raw = bytes(dec[0]) if isinstance(dec, tuple) else bytes(dec)
        except Exception as exc:
            raise FrameCodecError(f"RS decode failed block {i}: {exc}") from exc
        out += raw
    return bytes(out[:payload_len])


def _render_frame(coded: bytes, total: int, idx: int, payload_len: int, data_crc: int, flags: int = 0) -> Image.Image:
    usable = INNER_CAP - 4 * CORNER * CORNER
    if len(coded) > usable - HEADER_AREA:
        raise FrameCodecError("Coded payload too large for frame")
    stream = np.zeros(usable, dtype=np.uint8)
    header = _pack_header(total, idx, payload_len, data_crc, flags)
    stream[:HEADER_AREA] = np.frombuffer(header * HEADER_REPEAT, dtype=np.uint8)
    stream[HEADER_AREA:HEADER_AREA + len(coded)] = np.frombuffer(coded, dtype=np.uint8)
    inner = np.zeros((INNER_H, INNER_W), dtype=np.uint8)
    inner[_corner_mask()] = stream
    inner[:CORNER, :CORNER] = 255
    inner[:CORNER, -CORNER:] = 255
    inner[-CORNER:, :CORNER] = 255
    inner[-CORNER:, -CORNER:] = 255
    canvas = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    canvas[BORDER:BORDER + INNER_H, BORDER:BORDER + INNER_W] = inner
    return Image.fromarray(canvas, mode="L")


def _parse_frame(img: Image.Image) -> tuple[dict, bytes]:
    if img.size != (WIDTH, HEIGHT):
        raise FrameCodecError(f"Bad frame size {img.size}, expected {(WIDTH, HEIGHT)}")
    arr = np.asarray(img.convert("L"))
    inner = arr[BORDER:BORDER + INNER_H, BORDER:BORDER + INNER_W]
    for ys, xs in ((0, 0), (0, 1), (1, 0), (1, 1)):
        y = slice(ys * (INNER_H - CORNER), ys * (INNER_H - CORNER) + CORNER)
        x = slice(xs * (INNER_W - CORNER), xs * (INNER_W - CORNER) + CORNER)
        if int(np.mean(inner[y, x])) < 200:
            raise FrameCodecError("Corner marker missing/corrupt — frame misaligned?")
    mask = _corner_mask()
    flat_full = inner.reshape(-1)
    mask_flat = mask.reshape(-1)
    stream = flat_full[mask_flat]
    header = _vote_header(bytes(stream[:HEADER_AREA]))
    meta = _unpack_header(header)
    available = len(stream) - HEADER_AREA
    if meta["flags"] == 1:
        coded_len = meta["payload_len"]
        if coded_len > available:
            log.warning("Parity frame %s claims %d bytes but only %d available — reading remainder (compat)",
                        meta["idx"], coded_len, available)
            coded_len = (available // CODE_BLOCK) * CODE_BLOCK
    else:
        coded_len = ((meta["payload_len"] + DATA_BLOCK - 1) // DATA_BLOCK) * CODE_BLOCK if meta["payload_len"] else 0
    coded = bytes(stream[HEADER_AREA:HEADER_AREA + coded_len])
    if len(coded) < coded_len:
        raise FrameCodecError("Truncated coded payload")
    return meta, coded


def encode_bytes(data: bytes, nsym: int = 16) -> list[tuple[Image.Image, dict]]:
    chunks = [data[i:i + DATA_PER_FRAME] for i in range(0, len(data), DATA_PER_FRAME)] or [b""]
    data_frames: list[tuple[Image.Image, dict]] = []
    for idx, chunk in enumerate(chunks):
        coded = _rs_encode_chunk(chunk, nsym)
        meta = {"idx": idx, "payload_len": len(chunk), "data_crc": zlib.crc32(chunk) & 0xFFFFFFFF, "flags": 0}
        img = _render_frame(coded, 0, idx, len(chunk), meta["data_crc"], 0)
        data_frames.append((img, {"coded": coded, **meta}))
    ndata = len(data_frames)
    frames: list[tuple[Image.Image, dict]] = []
    parity_count = 0
    for gstart in range(0, ndata, PARITY_GROUP):
        group = data_frames[gstart:gstart + PARITY_GROUP]
        for img, m in group:
            frames.append((img, m))
        if len(group) == PARITY_GROUP:
            maxlen = max(len(m["coded"]) for _, m in group)
            acc = bytearray(maxlen)
            for _, m in group:
                c = m["coded"]
                for i in range(len(c)):
                    acc[i] ^= c[i]
            parity = bytes(acc)
            p_crc = zlib.crc32(parity) & 0xFFFFFFFF
            p_img = _render_frame(parity, 0, gstart, maxlen, p_crc, 1)
            frames.append((p_img, {"coded": parity, "idx": gstart, "payload_len": maxlen, "data_crc": p_crc, "flags": 1}))
            parity_count += 1
    total = len(frames)
    fixed: list[tuple[Image.Image, dict]] = []
    data_idx = 0
    for img, m in frames:
        if m["flags"] == 0:
            coded = m["coded"]
            fixed.append((_render_frame(coded, total, data_idx, m["payload_len"], m["data_crc"], 0), {**m, "idx": data_idx}))
            data_idx += 1
        else:
            fixed.append((_render_frame(m["coded"], total, m["idx"], m["payload_len"], m["data_crc"], 1), m))
    log.info("Encoded %d bytes -> %d frames (%d data + %d parity)", len(data), total, ndata, parity_count)
    return fixed


def _data_len_for(idx: int, expected: int, total_bytes: int | None) -> int | None:
    if idx < expected - 1:
        return DATA_PER_FRAME
    if total_bytes is not None:
        return total_bytes - DATA_PER_FRAME * (expected - 1)
    return None


def decode_images(images: list[Image.Image], nsym: int = 16, total_bytes: int | None = None,
                  expected_data: int | None = None) -> bytes:
    metas: list[dict] = []
    coded_list: list[bytes] = []
    for img in images:
        meta, coded = _parse_frame(img)
        metas.append(meta)
        coded_list.append(coded)
    data_items = [(m, c) for m, c in zip(metas, coded_list) if m["flags"] == 0]
    parity_items = [(m, c) for m, c in zip(metas, coded_list) if m["flags"] == 1]
    if not data_items:
        raise FrameCodecError("No data frames found")
    total_declared = metas[0]["total"]
    data_items.sort(key=lambda t: t[0]["idx"])
    expected = max(m["idx"] for m, _ in data_items) + 1
    if expected_data and expected_data > expected:
        expected = expected_data
    elif total_bytes:
        expected = max(expected, (total_bytes + DATA_PER_FRAME - 1) // DATA_PER_FRAME)
    by_idx = {m["idx"]: (m, c) for m, c in data_items}
    full_groups = [g for g in range(0, expected, PARITY_GROUP)
                   if len(list(range(g, min(g + PARITY_GROUP, expected)))) == PARITY_GROUP]
    mapped: dict[int, tuple[dict, bytes]] = {}
    unmapped: list[tuple[dict, bytes]] = []
    for pm, pc in parity_items:
        g = pm["idx"]
        if g in full_groups:
            mapped[g] = (pm, pc)
        else:
            unmapped.append((pm, pc))
    for i, g in enumerate(full_groups):
        if g not in mapped and i < len(unmapped):
            mapped[g] = unmapped[i]
    if len(unmapped) > len(full_groups):
        log.warning("Ignoring %d surplus parity frames", len(unmapped) - len(full_groups))
    for g in sorted(mapped):
        pm, pc = mapped[g]
        members = list(range(g, min(g + PARITY_GROUP, expected)))
        miss = [i for i in members if i not in by_idx]
        if len(miss) != 1:
            continue
        target = miss[0]
        maxlen = len(pc)
        for i in members:
            if i in by_idx:
                maxlen = max(maxlen, len(by_idx[i][1]))
        acc = bytearray(pc[:maxlen] + b"\x00" * (maxlen - len(pc)))
        for i in members:
            if i in by_idx:
                cc = by_idx[i][1]
                for k in range(len(cc)):
                    acc[k] ^= cc[k]
        hint = _data_len_for(target, expected, total_bytes)
        rec_meta = {"idx": target, "payload_len": hint or 0, "data_crc": 0, "flags": 0, "recovered": True}
        by_idx[target] = (rec_meta, bytes(acc))
        log.info("Recovered data frame %d via parity of group starting at %d", target, g)
    missing = [i for i in range(expected) if i not in by_idx]
    if missing:
        raise FrameCodecError(f"Missing data frames (no parity recovery): {missing}")
    if total_declared and len(images) < total_declared * 0.5:
        log.warning("Many frames missing: got %d, declared %d", len(images), total_declared)
    out = bytearray()
    for i in range(expected):
        m, c = by_idx[i]
        payload_len = m["payload_len"]
        if not payload_len:
            hint = _data_len_for(i, expected, total_bytes)
            payload_len = hint if hint else (len(c) // CODE_BLOCK) * DATA_BLOCK
        raw = _rs_decode_chunk(c, payload_len, nsym)
        if m.get("data_crc") and (zlib.crc32(raw) & 0xFFFFFFFF) != m["data_crc"]:
            raise FrameCodecError(f"CRC mismatch frame {i} — data corrupt beyond RS repair")
        out += raw
    if total_bytes and len(out) > total_bytes:
        out = out[:total_bytes]
    return bytes(out)


def save_frames(frames: list[tuple[Image.Image, dict]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    seq = 0
    for img, m in frames:
        p = out_dir / f"frame_{seq:05d}.png"
        img.save(p, optimize=False)
        paths.append(p)
        seq += 1
    return paths


def load_frames(frame_paths: list[Path]) -> list[Image.Image]:
    return [Image.open(p) for p in sorted(frame_paths)]
