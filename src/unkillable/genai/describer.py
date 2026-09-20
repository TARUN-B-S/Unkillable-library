import base64
from pathlib import Path
from typing import Optional

import requests

from unkillable.config import load_config, section
from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)

DEFAULT_PROVIDER = "ollama"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:31b-cloud"


class DescriberError(RuntimeError):
    pass


class Describer:
    def __init__(
        self,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        config_path: Optional[str] = None,
    ) -> None:
        cfg = section(load_config(config_path), "genai")
        self.provider = provider or str(cfg.get("provider") or DEFAULT_PROVIDER)
        self.base_url = str(base_url or cfg.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self.model = str(model or cfg.get("model") or DEFAULT_MODEL)
        self.timeout = timeout if timeout is not None else int(cfg.get("timeout") or 60)

    def describe(
        self,
        image_path: Path | str,
        prompt: str = "Look carefully at this security camera image. Note anything abnormal, suspicious, unusual, or otherwise notable — objects, behavior, or details worth flagging. If nothing stands out, say so briefly.",
    ) -> str:
        p = Path(image_path)
        if not p.exists():
            raise DescriberError(f"Image not found: {p}")
        if self.provider == "ollama":
            return self._ollama_describe(p, prompt)
        raise DescriberError(f"Unknown provider: {self.provider}")

    def _ollama_describe(self, image_path: Path, prompt: str) -> str:
        try:
            data = base64.b64encode(image_path.read_bytes()).decode()
        except OSError as exc:
            raise DescriberError(f"Failed to read image: {exc}") from exc
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "images": [data], "stream": False}
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            if not text:
                raise DescriberError("Empty response from Ollama")
            log.info("Generated description for %s: %s", image_path.name, text[:80])
            return text
        except requests.ConnectionError as exc:
            log.warning("Ollama not reachable at %s: %s — using fallback", self.base_url, exc)
            return self._fallback_description(image_path)
        except requests.RequestException as exc:
            log.error("Ollama request failed: %s", exc)
            raise DescriberError(f"Ollama error: {exc}") from exc

    def _fallback_description(self, path: Path) -> str:
        return f"Security camera frame from {path.name} — motion event captured (Ollama offline, fallback description)."

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
