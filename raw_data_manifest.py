"""Helpers for working with collected raw-data manifests."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


MANIFEST_DIR = Path("data/raw/manifest")


@dataclass
class ManifestEntry:
    dataset: str
    path: Path
    source: str
    metadata: Dict


class RawDataManifest:
    """Convenience wrapper around a raw data manifest."""

    def __init__(self, manifest_path: Path):
        self.path = manifest_path
        with manifest_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self.season: int = payload.get("season")
        self.week: int = payload.get("week")
        self.generated_at: str = payload.get("generated_at")
        self._entries_by_dataset: Dict[str, List[ManifestEntry]] = defaultdict(list)

        for artifact in payload.get("artifacts", []):
            dataset = artifact.get("dataset")
            path_str = artifact.get("path")
            if not dataset or not path_str:
                continue
            path = Path(path_str)
            entry = ManifestEntry(
                dataset=dataset,
                path=path,
                source=artifact.get("source", ""),
                metadata=artifact.get("metadata", {}),
            )
            self._entries_by_dataset[dataset].append(entry)

    @classmethod
    def from_latest(cls) -> Optional["RawDataManifest"]:
        latest_path = MANIFEST_DIR / "latest.json"
        if not latest_path.exists():
            return None
        return cls(latest_path)

    def datasets(self) -> Iterable[str]:
        return self._entries_by_dataset.keys()

    def entries(self, dataset: str) -> List[ManifestEntry]:
        return self._entries_by_dataset.get(dataset, [])

    def _select_entry(self, dataset: str, identifier: Optional[str]) -> Optional[ManifestEntry]:
        candidates = self.entries(dataset)
        if not candidates:
            return None

        if identifier is None:
            if len(candidates) > 1:
                raise ValueError(f"Multiple entries for dataset '{dataset}' – specify an identifier")
            return candidates[0]

        target = str(identifier)
        for entry in candidates:
            if entry.path.stem == target:
                return entry

        return None

    def load_json(self, dataset: str, identifier: Optional[str] = None) -> Optional[Dict]:
        entry = self._select_entry(dataset, identifier)
        if not entry:
            return None
        with entry.path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def load_dataframe(self, dataset: str, identifier: Optional[str] = None) -> Optional[pd.DataFrame]:
        entry = self._select_entry(dataset, identifier)
        if not entry:
            return None
        return pd.read_csv(entry.path)

    def list_identifiers(self, dataset: str) -> List[str]:
        return [entry.path.stem for entry in self.entries(dataset)]
