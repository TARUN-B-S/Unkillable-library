# 🏛️ THE UNKILLABLE LIBRARY

## A Camera System That Cannot Be Killed, Hacked, or Lost

---

## What Is This?

A home security camera system that:

1. **Only records when something moves** — saves disk space, catches what matters
2. **Lets you search footage in plain English** — type "when did the red car leave?" and find it
3. **Backs itself up onto plain paper** — QR codes you can print, file in a cabinet, and restore with a webcam
4. **Runs 100% locally** — no cloud, no server, no subscription, no company that can go bankrupt or get hacked

---

## The Problem It Solves

Traditional security cameras have three fatal flaws:

| Flaw | What Happens | Our Solution |
|------|-------------|--------------|
| **Cloud dependency** | Company goes down, your footage is gone | Everything runs on your hardware |
| **No smart search** | You scrub through hours of video manually | AI lets you type plain English queries |
| **No physical backup** | If the NVR is stolen/destroyed, footage is lost | Paper QR backup survives anything |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    YOUR HOUSE                            │
│                                                          │
│  ┌──────────┐      ┌──────────────────────────────────┐ │
│  │  IP      │ RTSP │         FRIGATE NVR              │ │
│  │  Camera  │─────►│                                  │ │
│  │ (any $30 │      │  ┌────────────┐ ┌──────────────┐ │ │
│  │  camera) │      │  │  MOTION    │ │  AI OBJECT   │ │ │
│  └──────────┘      │  │  DETECTOR  │ │  DETECTOR    │ │ │
│                     │  └─────┬──────┘ └──────┬───────┘ │ │
│                     │        │                │         │ │
│                     │        ▼                ▼         │ │
│                     │  ┌─────────────────────────────┐ │ │
│                     │  │     RECORDING ENGINE         │ │ │
│                     │  │  Records ONLY when motion    │ │ │
│                     │  │  detected + object tracked   │ │ │
│                     │  └─────────────┬───────────────┘ │ │
│                     │                │                  │ │
│                     │  ┌─────────────▼───────────────┐ │ │
│                     │  │     LOCAL DISK STORAGE      │ │ │
│                     │  │  /media/frigate/             │ │ │
│                     │  │  - video clips (.mp4)        │ │ │
│                     │  │  - thumbnails (images)       │ │ │
│                     │  │  - embeddings (vectors)      │ │ │
│                     │  └─────────────┬───────────────┘ │ │
│                     │                │                  │ │
│                     │  ┌─────────────▼───────────────┐ │ │
│                     │  │     SEMANTIC SEARCH         │ │ │
│                     │  │  Jina CLIP model (local)    │ │ │
│                     │  │  Type: "red car leaving"    │ │ │
│                     │  │  Returns: matching clips    │ │ │
│                     │  └─────────────┬───────────────┘ │ │
│                     │                │                  │ │
│                     │  ┌─────────────▼───────────────┐ │ │
│                     │  │     PAPER BACKUP            │ │ │
│                     │  │  qr-backup → PDF with QRs   │ │ │
│                     │  │  Print → File in cabinet     │ │ │
│                     │  │  Restore → Webcam + zbar     │ │ │
│                     │  └─────────────────────────────┘ │ │
│                     └──────────────────────────────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────────┐│
│  │              WEB UI (localhost:5000)                  ││
│  │  • Live camera view                                  ││
│  │  • Browse recorded events                            ││
│  │  • Search: "person at door at 3am"                   ││
│  │  • View AI-generated descriptions                    ││
│  │  • Download clips                                    ││
│  └──────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
          │
          │ (nothing leaves your house)
          │
          ▼
        ╳ NO CLOUD
        ╳ NO SERVER  
        ╳ NO INTERNET REQUIRED
```

---

## How Each Part Works

### Part 1: Motion-Only Recording

**The Problem:** Security cameras that record 24/7 waste disk space. A 4K camera recording continuously fills a 1TB drive in ~3 days.

**The Solution:** Frigate's `mode: motion` setting.

```
Normal camera:     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (24/7 recording)
                   ←─────────── 24 hours ──────────────→

Unkillable Library: ░░░░░░░░░░░▓▓▓▓░░░░░░░▓▓▓░░░░░░░░  (motion only)
                   ←── empty ──→←motion→
```

**How motion detection works:**

1. Frigate compares each frame to the previous frame
2. If pixels changed enough (configurable threshold), it flags "motion detected"
3. Only then does it start recording
4. Recording continues until motion stops + a small buffer period

**Config:**
```yaml
record:
  retain:
    mode: motion      # Only record on motion
    days: 30          # Keep for 30 days, then auto-delete
```

**Result:** Instead of 24/7 footage, you get maybe 2-4 hours of actual events per day. Your disk lasts months instead of days.

---

### Part 2: AI Object Detection

**The Problem:** A tree branch waving in the wind triggers "motion" but isn't a threat.

**The Solution:** After motion is detected, Frigate runs an AI model to answer: *"Is this a person, car, dog, etc.?"*

```
Motion detected → Run AI classifier → Person? Car? Cat? Nothing?
                                         │
                                    If recognized → Create event
                                    If not → Discard (was just a tree)
```

**Two hardware options:**

| Hardware | Cost | Speed | Accuracy |
|----------|------|-------|----------|
| CPU only | $0 | 1-2 fps | Good |
| Google Coral USB | ~$30 | 100+ fps | Excellent |
| Intel NPU / GPU | $0 (if you have it) | 30+ fps | Excellent |

**What objects can it detect?**
- `person` — humans
- `car`, `truck`, `bus` — vehicles
- `dog`, `cat`, `bird` — animals
- `bicycle`, `motorcycle` — personal transport
- And many more (Frigate supports 60+ object types)

**Config:**
```yaml
objects:
  track:
    - person
    - car
    - dog
```

---

### Part 3: Semantic Search (Plain English)

**The Problem:** You know a red car was in your driveway last Tuesday, but you don't know when exactly. Scrolling through 30 days of footage is painful.

**The Solution:** Frigate's Semantic Search using CLIP embeddings.

**How it works — step by step:**

```
STEP 1: When an object is detected, Frigate saves a thumbnail
┌──────────────┐
│ [thumbnail   │
│  of person]  │
└──────────────┘

STEP 2: The CLIP model converts the image into a vector (list of numbers)
┌──────────────┐
│ [0.23, -0.81,│
│  0.44, 0.12, │
│  -0.55, ...] │  ← 512 numbers that "describe" this image
└──────────────┘

STEP 3: This vector is stored in a database with a timestamp
┌──────────────┐
│ ID: 8472     │
│ Time: 3:42am │
│ Camera: front│
│ Vector: [...]│
└──────────────┘

STEP 4: When you search, your TEXT is converted to the SAME kind of vector
┌──────────────┐
│ "red car"    │ → CLIP → [0.21, -0.79, 0.42, ...]
└──────────────┘

STEP 5: The system finds vectors closest to yours
┌──────────────┐
│ Your query   │──┐
│ [0.21, ...]  │  ├── cosine similarity ──► TOP MATCHES
│              │  │
│ Thumbnail    │──┘
│ [0.23, ...]  │
└──────────────┘
```

**What you can search for:**

| Query | What Frigate Finds |
|-------|--------------------|
| "red car" | Any car thumbnail that looks red |
| "person at door" | Thumbnails where a person is near a door |
| "delivery truck" | Vehicles that look like delivery trucks |
| "dog in yard" | Dog thumbnails from yard cameras |
| "person running" | Motion patterns suggesting running |

**Config:**
```yaml
semantic_search:
  enabled: True
  model: "jinav1"      # Jina CLIP — runs locally
  model_size: small    # CPU-friendly quantized version
```

**Important:** This runs 100% on your hardware. No data is sent anywhere. The Jina model is downloaded once and runs offline.

---

### Part 4: AI-Generated Descriptions

**The Problem:** Semantic search on thumbnails works for visual similarity, but doesn't understand *what's happening*.

**The Solution:** Frigate sends object thumbnails to a local LLM (via Ollama) to generate text descriptions.

```
Thumbnail: [person walking toward door]
         ↓
Local LLM: "Person approaching front door, appears to be 
            carrying a package, moving from street to porch"
         ↓
Stored in database alongside the thumbnail
         ↓
Now you can search by DESCRIPTION, not just image
```

**Config:**
```yaml
genai:
  provider: ollama
  base_url: http://localhost:11434
  model: qwen3-vl:8b-instruct   # Local vision-language model
```

**This means you can search:**
- "Who was at my door last night?" → finds the clip with description "person at door"
- "When did the delivery arrive?" → matches "delivery person with package"
- "What happened at 3am?" → shows all events with descriptions

---

### Part 5: Paper QR Backup

**The Problem:** If someone steals your NVR, or your hard drive dies, or a fire destroys your equipment — your evidence is gone.

**The Solution:** `qr-backup` encodes your data into printable QR code PDFs.

**How it works:**

```
                    ENCODING (Backup)
┌──────────────────┐
│ Event database    │ ──┐
│ Thumbnails        │ ──┼──► Compress ──► Split ──► QR codes ──► PDF
│ Search embeddings │ ──┘
└──────────────────┘

                    DECODING (Restore)
┌──────────────────┐
│ Printed PDF pages │ ──┐
│ (or photos of     │ ──┼──► Webcam/Scanner ──► Decode QRs ──► Reassemble
│  paper backup)    │ ──┘
└──────────────────┘
```

**What gets backed up:**

| Data | Why |
|------|-----|
| Event metadata (time, camera, object type) | Know WHAT happened and WHEN |
| Thumbnails | Visual proof of what was detected |
| Search embeddings | So semantic search still works after restore |
| Config | So the system can restart from scratch |

**What does NOT get backed up (too large):**

| Data | Why not |
|------|---------|
| Video clips | Too large for QR (would be thousands of pages) |
| Full-resolution images | Thumbnails are sufficient for search |

**Storage capacity:**

| Mode | Per Page | One Day's Metadata |
|------|----------|-------------------|
| Default | ~3KB | 2-5 pages |
| Dense | ~130KB | <1 page |

**A typical month of metadata = 30-150 pages of paper.**

**Redundancy:** The tool adds 30% redundancy by default — you can lose up to 30% of pages and still restore everything.

---

## Complete Data Flow

Here's what happens from start to finish:

```
1. CAMERA sees motion
   └─► Sends RTSP stream to Frigate

2. FRIGATE detects motion (pixel change)
   └─► Starts recording buffer

3. FRIGATE runs AI classifier
   └─► "That's a person" or "That's a car"

4. FRIGATE saves event
   ├─► Video clip (.mp4)
   ├─► Thumbnail image
   ├─► CLIP embedding (for search)
   └─► GenAI description (if enabled)

5. USER opens web UI (localhost:5000)
   └─► Can browse events, search, view clips

6. USER searches "red car leaving"
   ├─► Converts text to CLIP vector
   ├─► Finds closest matching thumbnails
   └─► Returns matching video clips

7. DAILY BACKUP (automated)
   ├─► Exports metadata + thumbnails + embeddings
   ├─► Compresses into .tar.gz
   ├─► Encodes to QR codes
   ├─► Generates PDF
   └─► User prints and files in cabinet

8. DISASTER RECOVERY
   ├─► Print photos of paper backup at 600dpi
   ├─► Decode QR codes with webcam
   ├─► Rebuild database
   └─► System is back online with searchable history
```

---

## Hardware Requirements

### Minimum (1-2 cameras)

| Component | Spec | Cost |
|-----------|------|------|
| Computer | Raspberry Pi 5 (8GB) or any old PC with 8GB RAM | $80-150 |
| Storage | 256GB SD card or SSD | $20-40 |
| Camera | Any RTSP IP camera (Reolink, Amcrest, etc.) | $30-80 |
| Coral TPU (optional) | Google Coral USB Accelerator | $30 |
| **Total** | | **$160-300** |

### Recommended (4-8 cameras)

| Component | Spec | Cost |
|-----------|------|------|
| Computer | Intel N150 NUC or Mini PC (16GB RAM) | $150-250 |
| Storage | 1TB SSD + 4TB HDD (for recordings) | $80-120 |
| Cameras | 4x RTSP IP cameras | $120-320 |
| Coral TPU | Google Coral USB | $30 |
| **Total** | | **$380-710** |

### Power Consumption

| Setup | Idle Power | Notes |
|-------|------------|-------|
| Raspberry Pi 5 | ~5W | Very efficient |
| Intel N150 NUC | ~6W | Great balance |
| Old desktop | ~30-50W | Works but costs more to run |

---

## What You Get

### The Web Interface (localhost:5000)

```
┌─────────────────────────────────────────────────────────┐
│  THE UNKILLABLE LIBRARY                        Settings  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐  ┌──────────────────────────────┐ │
│  │                  │  │  RECENT EVENTS                │ │
│  │   LIVE CAMERA    │  │                              │ │
│  │      VIEW        │  │  ▶ 3:42am — Person — Front   │ │
│  │                  │  │  ▶ 3:15am — Car — Driveway    │ │
│  │                  │  │  ▶ 2:58am — Dog — Backyard    │ │
│  └──────────────────┘  │  ▶ 2:30am — Person — Gate     │ │
│                         └──────────────────────────────┘ │
│                                                          │
│  🔍 Search: "red car leaving at night"                  │
│  ─────────────────────────────────────────               │
│  ┌──────────────────────────────────────────────────┐   │
│  │  RESULTS:                                        │   │
│  │  1. 3:42am — Red sedan exiting driveway          │   │
│  │     [thumbnail] [▶ Play] [Description]           │   │
│  │  2. 2:15am — Red vehicle passing on street       │   │
│  │     [thumbnail] [▶ Play] [Description]           │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Paper Backup

```
┌─────────────────────────────────────────┐
│  THE UNKILLABLE LIBRARY — BACKUP        │
│  Date: 2026-08-27                        │
│  Camera: front_gate                      │
│  Events: 47                              │
│                                          │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │ QR  │ │ QR  │ │ QR  │ │ QR  │       │
│  │  1  │ │  2  │ │  3  │ │  4  │       │
│  └─────┘ └─────┘ └─────┘ └─────┘       │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │ QR  │ │ QR  │ │ QR  │ │ QR  │       │
│  │  5  │ │  6  │ │  7  │ │  8  │       │
│  └─────┘ └─────┘ └─────┘ └─────┘       │
│                                          │
│  RESTORE: Scan with webcam using         │
│  qr-backup --restore                     │
│  Or: zbarimg --raw *.jpg > backup.tar.gz │
└─────────────────────────────────────────┘
```

---

## Security Properties

| Threat | Traditional Cloud Camera | The Unkillable Library |
|--------|------------------------|----------------------|
| Company goes bankrupt | Footage lost forever | You own everything |
| Cloud account hacked | Attacker sees your home | Nothing to hack (no cloud) |
| Internet outage | Camera stops working | Records locally, no internet needed |
| NVR stolen | All footage gone | Paper backup in filing cabinet |
| Hard drive dies | Footage lost | Paper backup survives |
| Firmware update bricked | Camera useless | Open source, fix it yourself |
| Subscription price hike | Pay or lose features | $0/month forever |
| Government data request | Company may comply | No company to request from |

---

## What's Next

This document covers the **concept and architecture**. To actually build it:

1. **Connect a camera** — get RTSP URL from your camera's settings
2. **Start Frigate** — `docker compose up -d` in the unkillable-library folder
3. **Test recording** — walk in front of the camera, verify it records
4. **Test search** — open localhost:5000, type a search query
5. **Set up paper backup** — install qr-backup, create automation script

Each of these steps can be done one at a time, testing as you go.

---

## Files Created

```
unkillable-library/
├── docker-compose.yml        # Tells Docker to run Frigate
├── config/
│   └── config.yml            # Frigate's configuration
├── storage/                  # Where recordings are saved
└── DOCUMENTATION.md          # This file
```

---

*Built with: Frigate NVR (open source), Jina CLIP (open weights), qr-backup (open source)*
*Total cost: ~$150-300 one-time. $0/month. 100% local. 100% yours.*
