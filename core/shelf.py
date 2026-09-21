"""Shelf persistence. stdlib sqlite3, same JSON shapes the API always returned."""
import json
import sqlite3
from datetime import datetime

from .config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  shelf TEXT, box TEXT, item_name TEXT, email TEXT,
  added_at TEXT, latest_json TEXT,
  PRIMARY KEY (shelf, box)
);
CREATE TABLE IF NOT EXISTS scans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  shelf TEXT, box TEXT, category TEXT,
  confidence REAL, shelf_life TEXT, ts TEXT
);
"""


class ShelfStore:
    def __init__(self, path: str = DB_PATH):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(_SCHEMA)

    def add(self, shelf: str, box: str, item_name: str, email: str | None) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.db.execute(
            "INSERT OR REPLACE INTO items VALUES (?,?,?,?,?,?)",
            (shelf, box, item_name, email, now, None),
        )
        self.db.execute("DELETE FROM scans WHERE shelf=? AND box=?", (shelf, box))
        self.db.commit()
        return f"{shelf}_{box}"

    def record_scan(self, shelf: str, box: str, result: dict) -> None:
        ts = result["timestamp"] if "timestamp" in result else datetime.now().strftime("%Y-%m-%d %H:%M")
        self.db.execute(
            "INSERT INTO scans (shelf, box, category, confidence, shelf_life, ts)"
            " VALUES (?,?,?,?,?,?)",
            (shelf, box, result["category"], result["confidence"], result["shelf_life"], ts),
        )
        self.db.execute(
            "UPDATE items SET latest_json=? WHERE shelf=? AND box=?",
            (json.dumps({**result, "timestamp": ts}), shelf, box),
        )
        self.db.commit()

    def get_all(self) -> dict:
        items = {
            f"{r['shelf']}_{r['box']}": {
                "shelf_number": r["shelf"], "box_number": r["box"],
                "item_name": r["item_name"], "email": r["email"],
                "added_at": r["added_at"], "scans": [],
                **({"latest": json.loads(r["latest_json"])} if r["latest_json"] else {}),
            }
            for r in self.db.execute("SELECT * FROM items")
        }
        for s in self.db.execute("SELECT * FROM scans ORDER BY id"):
            key = f"{s['shelf']}_{s['box']}"
            if key in items:
                items[key]["scans"].append({
                    "category": s["category"], "confidence": s["confidence"],
                    "shelf_life": s["shelf_life"], "timestamp": s["ts"],
                })
        return items

    def delete(self, shelf: str, box: str) -> bool:
        cur = self.db.execute("DELETE FROM items WHERE shelf=? AND box=?", (shelf, box))
        self.db.execute("DELETE FROM scans WHERE shelf=? AND box=?", (shelf, box))
        self.db.commit()
        return cur.rowcount > 0
