import re
import unicodedata
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class JD(BaseModel):
    job_id: str = "delivery_driver"
    title: str
    company: str
    employment_types: list[str] = Field(default_factory=list)
    shifts: list[str] = Field(default_factory=list)
    languages_supported: list[str] = Field(default_factory=lambda: ["es", "en"])
    service_areas: dict[str, list[str]] = Field(default_factory=dict)
    compensation: Optional[dict] = None
    perks: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    contact: Optional[str] = None
    body: str = ""

    def all_service_areas(self) -> list[str]:
        out: list[str] = []
        for cities in self.service_areas.values():
            out.extend(cities)
        return out

    def is_in_service_area(self, city: str) -> bool:
        if not city:
            return False
        target = _normalize(city)
        return any(_normalize(c) == target for c in self.all_service_areas())


_FRONTMATTER_RE = re.compile(r"^---\s*\n(?P<yaml>.*?)\n---\s*\n(?P<body>.*)$", re.DOTALL)


def _normalize(s: str) -> str:
    """Accent- and case-insensitive normalized form for fuzzy city matching."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip().lower()


def load_jd(path: Path) -> JD:
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(f"JD at {path} is missing required YAML frontmatter")
    meta = yaml.safe_load(m.group("yaml")) or {}
    return JD(**meta, body=m.group("body").strip())
