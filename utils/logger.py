from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                flat[f"{key}_{child_key}"] = child_value
        elif isinstance(value, (list, tuple)):
            for idx, child_value in enumerate(value):
                flat[f"{key}_{idx}"] = child_value
        else:
            flat[key] = value
    return flat


class ExperimentLogger:
    def __init__(self, log_dir: str | Path):
        self.log_dir = ensure_dir(log_dir)
        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self.csv_path = self.log_dir / "metrics.csv"
        self._csv_fieldnames: list[str] | None = None

    def log(self, metrics: dict[str, Any]) -> None:
        flat = flatten_metrics(metrics)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        if self._csv_fieldnames is None:
            self._csv_fieldnames = sorted(flat.keys())
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
                writer.writeheader()
                writer.writerow(flat)
            return

        missing = [field for field in flat.keys() if field not in self._csv_fieldnames]
        if missing:
            self._csv_fieldnames.extend(sorted(missing))
            existing_rows = list(self.read_csv_rows())
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow(row)
                writer.writerow(flat)
            return

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
            writer.writerow(flat)

    def read_csv_rows(self) -> list[dict[str, Any]]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def save_json(self, name: str, payload: dict[str, Any]) -> None:
        path = self.log_dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
