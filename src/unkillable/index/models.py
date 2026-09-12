from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class IndexEntry:
    """A single indexed frame from a video clip."""

    id: str
    timestamp: float  # seconds into the source clip
    source_clip: str  # path to the source video clip
    thumbnail_path: str  # path to the extracted keyframe image
    motion_score: float = 0.0
    labels: list[str] = field(default_factory=list)
    objects: list[dict] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    detection_confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "IndexEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def timestamp_str(self) -> str:
        """Human-readable timestamp like '01:23'."""
        mins = int(self.timestamp) // 60
        secs = int(self.timestamp) % 60
        return f"{mins:02d}:{secs:02d}"

    @property
    def thumbnail_path_obj(self) -> Path:
        return Path(self.thumbnail_path)
