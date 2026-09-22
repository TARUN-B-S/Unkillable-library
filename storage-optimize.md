# Storage Optimize — Grayscale Dense Frame Backup

## Goal
Daily encode entire `storage/` into N grayscale `1920x1080` PNGs in `backups/YYYY-MM-DD/`,
with FFmpeg compression, tarball staging, light Reed-Solomon correction + full detection.
Restore back to `storage/` losslessly. Optional verified delete of source.

## Decisions (locked)
- Grayscale `L` PNG, 1 byte/px, lossless only (never JPEG the backups).
- Space-saving: minimal framing, ~10% RS parity (some correction), 100% detection
  (CRC32 per 32KB chunk + CRC32 per frame + SHA256 manifest).
- FFmpeg lossy: H.264 CRF-28 `veryfast` for video, JPEG q70/WebP q70 for images.
- Scope: whole `storage/` (clips, recordings, thumbnails, embeddings, events).
- Trigger: CLI only (`backup-frames` run once/day by user cron/systemd).
- Delete: opt-in `--delete-source`, only after `--verify` round-trip passes.

## Capacity
- Raw: 1920*1080 = 2,073,600 B ≈ 2.0 MB/frame.
- Overhead: 40px quiet border + 4x24px corner squares + ~2KB header
  (magic `UKLB1`, backup date, total/idx, payload len, frame CRC).
- Net after 10% RS: ~1.7-1.8 MB/frame.
- Current storage ~32MB -> tarball ~20-25MB after CRF-28 -> ~12-15 PNGs/day.

## Encode pipeline (`storage/` -> `backups/YYYY-MM-DD/`)
1. Inventory `storage/`, collect file list + sha256 into manifest draft.
2. Stage to temp dir with FFmpeg compress:
   - video (mp4/mkv/avi/mov/webm): `ffmpeg -y -i in -c:v libx264 -crf 28 -preset veryfast -pix_fmt yuv420p -c:a aac -b:a 64k out.mp4`, fallback `-c copy`.
   - image (jpg/jpeg/png/webp/bmp): re-encode JPEG q70 / WebP q70 via FFmpeg or Pillow, keep original if smaller.
3. `tar -czf daily.tar.gz` from staged dir.
4. Split tarball into ~1.7MB frame payloads -> 255B blocks -> `reedsolo.RSCodec(nsym=16)` (~6%) + 1 parity frame per 10 data frames (~10% total).
5. Render each payload row-major into `1920x1080` `L` image via numpy+Pillow, stamp corners + header, save `frame_00000.png` + `manifest.json` (RS params, counts, tarball sha256, ffmpeg flags).

Modules:
- `src/unkillable/backup/frame_codec.py` — shard/RS/render/parse.
- `src/unkillable/backup/daily_backup.py` — inventory + FFmpeg compress + tar + orchestrate.
- `src/unkillable/backup/frame_restore.py` — decode + verify + extract.

## Decode pipeline (`backups/YYYY-MM-DD/` -> `storage/`)
1. Load PNGs sorted, crop border, locate corner markers, read header, CRC check.
2. RS-decode (corrects ~6-10% corrupt bytes/frame, detects rest), concat payloads.
3. Verify sha256 vs `manifest.json`, `tar -xzf` to target dir.
4. Any CRC/SHA mismatch -> abort with bad frame idx, no partial restore.

## CLI
- `unkillable backup-frames --source storage --dest backups [--date auto] [--crf 28] [--rs-nsym 16] [--verify] [--delete-source]`
- `unkillable restore-frames <date-dir> --output storage_restored [--verify-only]`
- `--delete-source` requires `--verify` pass, otherwise refuse.

## Deps / tests / docs
- Add `reedsolo` (+ existing `numpy`, `pillow`). Keep old QR-PDF code untouched.
- Tests `tests/test_frame_backup.py`: round-trip, 1% byte-flip corrected, 20% flip detected/fails, ffmpeg shrink check, delete-guard test.
- Risk: single-pixel bytes need bit-exact PNGs — never resize/recompress backups.
