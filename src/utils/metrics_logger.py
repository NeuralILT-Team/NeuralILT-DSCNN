"""
Simple CSV logger for tracking training metrics.
"""

import csv
import json
from pathlib import Path
from datetime import datetime


class MetricsLogger:
    """Logs metrics to CSV and can dump to JSON."""

    def __init__(self, log_dir, name="experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"{name}_metrics.csv"
        self.json_path = self.log_dir / f"{name}_metrics.json"
        self.history = []

        # If CSV already exists (e.g., resuming training), append to it
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            self._header_written = True
        else:
            self._header_written = False

    def log(self, metrics, step=None):
        entry = {"timestamp": datetime.now().isoformat()}
        if step is not None:
            entry["step"] = step
        entry.update(metrics)
        self.history.append(entry)
        self._write_csv(entry)

    def _write_csv(self, entry):
        mode = 'a' if self._header_written else 'w'
        with open(self.csv_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(entry)

    def save_json(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def print_epoch(self, epoch, metrics):
        parts = [f"Epoch {epoch:3d}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}: {v:.6f}")
            else:
                parts.append(f"{k}: {v}")
        print(" | ".join(parts))
