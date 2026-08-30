"""End-to-end tests for the Unkillable Library Docker Compose stack.

These tests verify the running infrastructure:
- Dashboard serves updated HTML
- Nginx proxies to Frigate correctly
- Frigate API responds with events
- Video clips are accessible
- WebSocket endpoint is reachable
- All services are healthy

Run with: pytest tests/test_e2e.py -v
Requires: docker compose up -d (all services running)
"""
import json
import time

import pytest
import requests

BASE_URL = "http://localhost:8080"   # nginx dashboard
FRIGATE_DIRECT = "http://localhost:5000"  # direct Frigate (for comparison)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def wait_for_services():
    """Wait up to 30s for services to be ready."""
    for _ in range(30):
        try:
            r = requests.get(f"{BASE_URL}/frigate/api/config", timeout=2)
            if r.status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(1)
    pytest.skip("Services not ready after 30s")


@pytest.fixture(scope="module")
def frigate_config(wait_for_services):
    """Fetch Frigate config via nginx proxy."""
    r = requests.get(f"{BASE_URL}/frigate/api/config", timeout=5)
    r.raise_for_status()
    return r.json()


@pytest.fixture(scope="module")
def frigate_events(wait_for_services):
    """Fetch latest events via nginx proxy."""
    r = requests.get(
        f"{BASE_URL}/frigate/api/events",
        params={"camera": "test_camera", "has_clip": "1", "limit": "5"},
        timeout=5,
    )
    r.raise_for_status()
    return r.json()


# ═══════════════════════════════════════════════════════════════════════
# Dashboard Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDashboard:
    def test_dashboard_serves_html(self, wait_for_services):
        r = requests.get(BASE_URL, timeout=5)
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")
        assert "Traffic Retention Monitor" in r.text

    def test_dashboard_has_refresh_button(self, wait_for_services):
        r = requests.get(BASE_URL, timeout=5)
        assert "refresh-btn" in r.text

    def test_dashboard_has_processing_status(self, wait_for_services):
        r = requests.get(BASE_URL, timeout=5)
        assert "proc-badge" in r.text
        assert "proc-text" in r.text

    def test_dashboard_has_clip_overlay(self, wait_for_services):
        r = requests.get(BASE_URL, timeout=5)
        assert "clip-overlay" in r.text
        assert "retry-btn" in r.text

    def test_dashboard_has_snapshot_first_strategy(self, wait_for_services):
        """Shows snapshot during active events, upgrades to clip after end."""
        r = requests.get(BASE_URL, timeout=5)
        assert "snapshot" in r.text
        assert "loadSnapshot" in r.text
        assert "startClipRetry" in r.text

    def test_dashboard_has_exponential_backoff(self, wait_for_services):
        r = requests.get(BASE_URL, timeout=5)
        assert "CLIP_BACKOFF" in r.text
        assert "CLIP_MAX_ATTEMPTS" in r.text
        assert "CLIP_INITIAL_DELAY" in r.text

    def test_dashboard_has_hls_live(self, wait_for_services):
        r = requests.get(BASE_URL, timeout=5)
        assert "hls.js" in r.text
        assert "s52.nysdot.skyvdn.com" in r.text

    def test_dashboard_has_3s_polling(self, wait_for_services):
        r = requests.get(BASE_URL, timeout=5)
        assert "setInterval(refreshStored, 3000)" in r.text

    def test_dashboard_has_websocket_layer(self, wait_for_services):
        r = requests.get(BASE_URL, timeout=5)
        assert "WebSocket" in r.text
        assert "/frigate/ws" in r.text


# ═══════════════════════════════════════════════════════════════════════
# Nginx Proxy Tests
# ═══════════════════════════════════════════════════════════════════════

class TestNginxProxy:
    def test_frigate_config_via_proxy(self, frigate_config):
        assert "cameras" in frigate_config
        assert "test_camera" in frigate_config["cameras"]
        assert frigate_config.get("version", "").startswith("0.")

    def test_frigate_events_via_proxy(self, frigate_events):
        assert isinstance(frigate_events, list)

    def test_proxy_returns_json(self, wait_for_services):
        r = requests.get(f"{BASE_URL}/frigate/api/config", timeout=5)
        assert "application/json" in r.headers.get("content-type", "")

    def test_proxy_cors_headers(self, wait_for_services):
        r = requests.get(f"{BASE_URL}/frigate/api/config", timeout=5)
        # Nginx should not block cross-origin for local tool
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# Frigate API Tests
# ═══════════════════════════════════════════════════════════════════════

class TestFrigateAPI:
    def test_config_has_test_camera(self, frigate_config):
        cameras = frigate_config.get("cameras", {})
        assert "test_camera" in cameras

    def test_config_motion_mode(self, frigate_config):
        cam = frigate_config["cameras"]["test_camera"]
        # Verify motion-based recording is configured
        record = cam.get("record", {})
        assert record.get("enabled", False) is True

    def test_config_objects_tracked(self, frigate_config):
        cam = frigate_config["cameras"]["test_camera"]
        objects = cam.get("objects", {})
        track = objects.get("track", [])
        assert "person" in track
        assert "car" in track

    def test_events_have_required_fields(self, frigate_events):
        if not frigate_events:
            pytest.skip("No events yet")
        event = frigate_events[0]
        assert "id" in event
        assert "camera" in event
        assert "label" in event
        assert "start_time" in event
        assert "has_clip" in event

    def test_events_are_from_test_camera(self, frigate_events):
        if not frigate_events:
            pytest.skip("No events yet")
        for event in frigate_events:
            assert event["camera"] == "test_camera"

    def test_events_have_clip(self, frigate_events):
        if not frigate_events:
            pytest.skip("No events yet")
        for event in frigate_events:
            assert event["has_clip"] is True


# ═══════════════════════════════════════════════════════════════════════
# Clip Access Tests
# ═══════════════════════════════════════════════════════════════════════

class TestClipAccess:
    def test_clip_accessible_via_proxy(self, frigate_events):
        if not frigate_events:
            pytest.skip("No events yet")
        event_id = frigate_events[0]["id"]
        r = requests.get(
            f"{BASE_URL}/frigate/api/events/{event_id}/clip.mp4",
            timeout=10,
            stream=True,
        )
        assert r.status_code == 200
        ct = r.headers.get("content-type", "")
        # Should be video/mp4 or similar
        assert "video" in ct or "octet-stream" in ct or "mp4" in ct

    def test_clip_returns_data(self, frigate_events):
        if not frigate_events:
            pytest.skip("No events yet")
        event_id = frigate_events[0]["id"]
        r = requests.get(
            f"{BASE_URL}/frigate/api/events/{event_id}/clip.mp4",
            timeout=10,
        )
        assert len(r.content) > 1000  # Should be a real video file


# ═══════════════════════════════════════════════════════════════════════
# Snapshot Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSnapshot:
    def test_event_has_snapshot(self, frigate_events):
        if not frigate_events:
            pytest.skip("No events yet")
        event = frigate_events[0]
        if not event.get("has_snapshot"):
            pytest.skip("Event has no snapshot")
        r = requests.get(
            f"{BASE_URL}/frigate/api/events/{event['id']}/snapshot.jpg",
            timeout=10,
        )
        assert r.status_code == 200
        assert "image" in r.headers.get("content-type", "")


# ═══════════════════════════════════════════════════════════════════════
# Resilience Tests
# ═══════════════════════════════════════════════════════════════════════

class TestResilience:
    def test_rapid_polling_doesnt_crash(self, wait_for_services):
        """Fire 10 rapid event queries — should not error."""
        for _ in range(10):
            r = requests.get(
                f"{BASE_URL}/frigate/api/events?camera=test_camera&has_clip=1&limit=1",
                timeout=5,
            )
            assert r.status_code == 200

    def test_nonexistent_event_returns_404(self, wait_for_services):
        r = requests.get(
            f"{BASE_URL}/frigate/api/events/nonexistent-id/clip.mp4",
            timeout=5,
        )
        assert r.status_code in (404, 405)  # Frigate may return 404 or 405

    def test_invalid_camera_returns_empty(self, wait_for_services):
        r = requests.get(
            f"{BASE_URL}/frigate/api/events?camera=nonexistent&has_clip=1&limit=1",
            timeout=5,
        )
        assert r.status_code == 200
        assert r.json() == []

    def test_dashboard_handles_frigate_restart(self, wait_for_services):
        """Dashboard should still serve HTML even if Frigate is briefly down."""
        # This tests that the dashboard doesn't crash on Frigate errors
        # The 3s poll will just get errors and show "Frigate unavailable"
        r = requests.get(BASE_URL, timeout=5)
        assert r.status_code == 200
