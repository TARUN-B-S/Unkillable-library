"""Tests for the Storyteller trio (`unkillable story`).

Run with the hash embedding mode so the suite is fast, deterministic, and
offline (same convention as the main semantic-search tests).
"""
import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path

os.environ.setdefault("UNKILLABLE_EMBEDDING_MODE", "hash")

from unkillable.index.models import IndexEntry
from unkillable.semantic.search import SemanticSearch
from unkillable.story import anomalies, diary, identities, ingest, narrator
from unkillable.story.runner import StoryRunner, parse_date

# ── helpers ───────────────────────────────────────────────────────────

def make_entry(entry_id, clip, camera, wall_time, objects=None, labels=None):
    objs = objects or []
    return IndexEntry(
        id=entry_id,
        timestamp=1.0,
        source_clip=clip,
        thumbnail_path=f"/nonexistent/{entry_id}.jpg",
        wall_time=wall_time,
        camera=camera,
        labels=labels or [o["label"] for o in objs if o.get("label")],
        objects=objs,
        counts={},
        embedding=[],
        text_embedding=[],
        metadata={},
        tags=[],
        detection_confidence=0.8,
    )


def car_obj(color="blue", bbox=(160, 160, 480, 480), size="medium", conf=0.9):
    return {
        "label": "car",
        "confidence": conf,
        "bbox": list(bbox),
        "area": max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]),
        "color_name": color,
        "size": size,
    }


def person_obj(bbox=(300, 60, 340, 520), size="medium"):
    return {
        "label": "person",
        "confidence": 0.95,
        "bbox": list(bbox),
        "area": max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]),
        "color_name": "red",
        "size": size,
    }


def make_event(event_id, camera, wall_time, objects, clip=None, n_frames=2):
    clip = clip or f"/clips/{event_id}.mp4"
    entries = [make_entry(f"{event_id}_f{i}", clip, camera, wall_time, objects) for i in range(n_frames)]
    return ingest.group_events(entries)[0]


def write_index(storage_root, entries):
    path = storage_root / "index" / "entries.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(e.to_dict(), ensure_ascii=False) + "\n" for e in entries)


class _FakeDescriber:
    def health_check(self):
        return False


# ── zones & fingerprints ──────────────────────────────────────────────

def test_zone_of_bins():
    assert identities.zone_of((40, 40, 160, 160)) == "NW"
    assert identities.zone_of((160, 160, 480, 480)) == "CC"
    assert identities.zone_of((480, 40, 600, 160)) == "NE"
    assert identities.zone_of(None) is None


def test_fingerprint_of_fields():
    fp = identities.fingerprint_of(car_obj())
    assert fp == ("car", "blue", "CC", "medium")
    assert identities.fingerprint_of({"confidence": 0.9}) is None


# ── Memory Palace ─────────────────────────────────────────────────────

def test_identity_same_car_merges_and_camera_splits():
    with tempfile.TemporaryDirectory() as d:
        store = identities.IdentityStore(Path(d) / "identities.json")
        t = datetime(2026, 9, 10, 8, 0).timestamp()
        obj = car_obj()
        id1 = store.assign("2026-09-10", "front", t, "evt1", obj)
        id2 = store.assign("2026-09-10", "front", t + 60, "evt1", obj)
        assert id1 == id2
        id3 = store.assign("2026-09-10", "back", t + 120, "evt2", obj)
        assert id3 != id1


def test_identity_splits_after_window():
    with tempfile.TemporaryDirectory() as d:
        store = identities.IdentityStore(Path(d) / "identities.json")
        t = datetime(2026, 9, 10, 8, 0).timestamp()
        obj = car_obj()
        id1 = store.assign("2026-09-10", "front", t, "evt1", obj)
        id2 = store.assign("2026-09-10", "front", t + 500, "evt2", obj)  # >180s
        assert id1 != id2


def test_identity_rerun_is_idempotent():
    with tempfile.TemporaryDirectory() as d:
        store = identities.IdentityStore(Path(d) / "identities.json")
        t = datetime(2026, 9, 10, 8, 0).timestamp()
        obj = car_obj()
        ident_id = store.assign("2026-09-10", "front", t, "evt1", obj)
        store.remove_events_on_date("2026-09-10")
        ident_id2 = store.assign("2026-09-10", "front", t, "evt1", obj)
        assert ident_id == ident_id2
        same = next(i for i in store.identities if i.id == ident_id)
        assert same.counts == 1  # re-assign doesn't duplicate sightings


def test_corrupt_identity_file_rebuilds():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "identities.json"
        path.write_text("{not json", encoding="utf-8")
        store = identities.IdentityStore(path)
        assert store.identities == []
        t = datetime(2026, 9, 10, 8, 0).timestamp()
        store.assign("2026-09-10", "front", t, "evt1", car_obj())
        store.save()
        assert path.exists()


# ── Anomaly Copilot ───────────────────────────────────────────────────

def test_anomalies_surge_missing_and_quiet():
    target = date(2026, 9, 10)
    entries = []
    # History: 5 prior days, 2 car frames at 08:00 and person frames at 09:00.
    for off in range(1, 6):
        day = datetime(2026, 9, 10 - off, 8, 0)
        entries.append(make_entry(f"h_car_{off}_1", "/clips/car.mp4", "front", day.timestamp(), [car_obj()]))
        entries.append(make_entry(f"h_car_{off}_2", "/clips/car.mp4", "front", day.timestamp() + 1, [car_obj()]))
    # Today: a big surge of cars at 08:00; persons never appear today (missing).
    for i in range(10):
        entries.append(make_entry(f"t_car_{i}", "/clips/car.mp4", "front", datetime(2026, 9, 10, 8, 0).timestamp() + i, [car_obj()]))
    # A label with no history must stay quiet (no false positives).
    entries.append(make_entry("t_dog", "/clips/dog.mp4", "front", datetime(2026, 9, 10, 12, 0).timestamp(), [person_obj()]))

    flags = anomalies.detect_anomalies(entries, target)
    kinds = {(a.camera, a.hour, a.label): a.kind for a in flags}
    assert kinds == {("front", 8, "car"): "surge"}


def test_anomaly_event_matching():
    a = anomalies.Anomaly("front", 8, "car", "surge", 10.0, 2.0, "surge msg")
    assert anomalies.event_anomalies([a], "front", 8, {"car", "person"}) == ["surge msg"]
    assert anomalies.event_anomalies([a], "front", 9, {"car"}) == []
    assert anomalies.event_anomalies([a], "front", 8, {"person"}) == []


# ── Narration ─────────────────────────────────────────────────────────

def test_template_is_deterministic():
    ev = make_event("evt1", "front", datetime(2026, 9, 10, 8, 5).timestamp(), [car_obj()])
    n = narrator.Narrator()
    assert n.template_line(ev) == n.template_line(ev)


def test_template_includes_identity_and_anomaly():
    ev = make_event("evt1", "front", datetime(2026, 9, 10, 8, 5).timestamp(), [car_obj()])
    line = narrator.Narrator().template_line(ev, identity_names=["blue car"], anomaly_msgs=["surge!"])
    assert "blue car" in line
    assert "surge!" in line
    assert "08:05" in line


def test_importance_ranks_person_over_far_car():
    n = narrator.Narrator()
    person = make_event("p", "front", datetime(2026, 9, 10, 8, 0).timestamp(), [person_obj()])
    car = make_event("c", "front", datetime(2026, 9, 10, 8, 0).timestamp(), [car_obj(bbox=(40, 40, 60, 60), size="tiny", conf=0.5)])
    assert n.importance(person) > n.importance(car)


# ── Ingest ────────────────────────────────────────────────────────────

def test_group_events_orders_and_dedups():
    t1 = datetime(2026, 9, 10, 8, 5).timestamp()
    t2 = datetime(2026, 9, 10, 9, 0).timestamp()
    entries = [
        make_entry("a_f1", "/clips/a.mp4", "front", t1, [car_obj()]),
        make_entry("a_f2", "/clips/a.mp4", "front", t1 + 1, [car_obj()]),
        make_entry("b_f1", "/clips/b.mp4", "back", t2, [person_obj()]),
    ]
    events = ingest.group_events(entries)
    assert [e.event_id for e in events] == ["a", "b"]
    assert events[0].camera == "front"
    assert len(events[0].objects) == 1  # deduped across frames


# ── Diary search ──────────────────────────────────────────────────────

def test_diary_search_roundtrip_hash():
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d) / "diary"
        idx = Path(d) / "diary_index.jsonl"
        dd.mkdir()
        (dd / "2026-09-10.txt").write_text(
            "# Nightwatch Diary — 2026-09-10\n"
            "- front: 2 events\n\n"
            "At 08:05, front: 1 car — blue car CC zone (2 frames) as blue car\n"
            "At 08:20, front: 1 person — red person C zone (1 frame)\n",
            encoding="utf-8",
        )
        assert diary.rebuild_index(dd, idx) == 2
        top = SemanticSearch(db_path=idx).search("blue car", top_k=1)
        assert top and "blue car" in top[0].metadata.get("text", "")


def test_rebuild_index_empty_diary():
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d) / "diary"
        idx = Path(d) / "diary_index.jsonl"
        dd.mkdir()
        assert diary.rebuild_index(dd, idx) == 0
        assert idx.exists()


# ── Runner (integration) ──────────────────────────────────────────────

def _seed_std_day(storage_root):
    day = date(2026, 9, 10)
    t = datetime(2026, 9, 10, 8, 5).timestamp()
    entries = []
    for i in range(3):  # prior-day baseline: cars at 08:00
        prior = datetime(2026, 9, 10 - (i + 1), 8, 0).timestamp()
        entries.append(make_entry(f"pcar{i}_1", "/clips/car.mp4", "front", prior, [car_obj()]))
        entries.append(make_entry(f"pcar{i}_2", "/clips/car.mp4", "front", prior + 1, [car_obj()]))
    entries.append(make_entry("d_car1", "/clips/car.mp4", "front", t, [car_obj()]))
    entries.append(make_entry("d_car2", "/clips/car.mp4", "front", t + 1, [car_obj()]))
    entries.append(make_entry("d_person", "/clips/walk.mp4", "front", datetime(2026, 9, 10, 8, 20).timestamp(), [person_obj()]))
    write_index(storage_root, entries)
    return day


def test_nightly_empty_day():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        runner = StoryRunner(storage_root=root, describer=_FakeDescriber())
        res = runner.nightly(date(2026, 9, 1))
        assert res.n_events == 0
        assert res.diary_path is not None and res.diary_path.exists()
        assert "No events recorded." in res.diary_path.read_text(encoding="utf-8")


def test_nightly_builds_diary_anomalies_and_identities():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        target = _seed_std_day(root)
        runner = StoryRunner(storage_root=root, describer=_FakeDescriber())
        res = runner.nightly(target)

        assert res.n_events == 2
        txt = res.diary_path.read_text(encoding="utf-8")
        assert "08:05" in txt and "08:20" in txt
        assert "- front: 2 events" in txt

        # Baseline: 3 prior days × 2 cars at 08:00 → mean 2; today 2 → quiet.
        assert (root / "story" / "anomalies.jsonl").exists()
        assert res.n_anomalies == 0

        # Two distinct detections → at least two identities persisted.
        assert res.n_identities >= 2
        ids = json.loads((root / "story" / "identities.json").read_text(encoding="utf-8"))
        assert isinstance(ids, list) and len(ids) >= 2

        # Diary indexed and searchable in hash mode.
        idx = root / "story" / "diary_index.jsonl"
        assert idx.exists()
        top = SemanticSearch(db_path=idx).search("person", top_k=1)
        assert top and "person" in top[0].metadata.get("text", "")


def test_nightly_is_idempotent():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        target = _seed_std_day(root)
        first = StoryRunner(storage_root=root, describer=_FakeDescriber()).nightly(target)
        text1 = first.diary_path.read_text(encoding="utf-8")
        ids1 = (root / "story" / "identities.json").read_text(encoding="utf-8")

        second = StoryRunner(storage_root=root, describer=_FakeDescriber()).nightly(target)
        text2 = second.diary_path.read_text(encoding="utf-8")
        ids2 = (root / "story" / "identities.json").read_text(encoding="utf-8")

        assert second.n_events == first.n_events
        assert second.n_identities == first.n_identities
        assert text1 == text2
        assert ids1 == ids2


def test_parse_date():
    assert parse_date("") == date.today()
    assert parse_date("2026-09-10") == date(2026, 9, 10)
    try:
        parse_date("not-a-date")
    except ValueError:
        return
    raise AssertionError("parse_date should reject garbage input")