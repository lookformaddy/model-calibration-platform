"""Hash-chained audit log.

Every registry event (register, promote, rollback, drift trigger) is
appended with the SHA-256 of the previous entry, so the log is
tamper-evident: editing or deleting an entry breaks the chain and
``verify()`` reports the first broken index.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _entry_hash(prev_hash: str, timestamp: str, event: str,
                payload: dict) -> str:
    body = json.dumps({"prev": prev_hash, "ts": timestamp, "event": event,
                       "payload": payload}, sort_keys=True).encode()
    return hashlib.sha256(body).hexdigest()


@dataclass
class AuditEntry:
    index: int
    timestamp: str
    event: str
    payload: dict
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> dict:
        return {"index": self.index, "timestamp": self.timestamp,
                "event": self.event, "payload": self.payload,
                "prev_hash": self.prev_hash, "entry_hash": self.entry_hash}


@dataclass
class AuditLog:
    entries: list[AuditEntry] = field(default_factory=list)

    def append(self, event: str, payload: dict | None = None) -> AuditEntry:
        prev = self.entries[-1].entry_hash if self.entries else "GENESIS"
        ts = datetime.now(timezone.utc).isoformat()
        entry = AuditEntry(
            index=len(self.entries), timestamp=ts, event=event,
            payload=payload or {}, prev_hash=prev,
            entry_hash=_entry_hash(prev, ts, event, payload or {}))
        self.entries.append(entry)
        return entry

    def verify(self) -> tuple[bool, int | None]:
        """Return (ok, first_broken_index or None)."""
        prev = "GENESIS"
        for entry in self.entries:
            if entry.prev_hash != prev:
                return False, entry.index
            expected = _entry_hash(entry.prev_hash, entry.timestamp,
                                   entry.event, entry.payload)
            if expected != entry.entry_hash:
                return False, entry.index
            prev = entry.entry_hash
        return True, None

    def to_dict(self) -> dict:
        return {"entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, d: dict) -> "AuditLog":
        log = cls()
        for e in d.get("entries", []):
            log.entries.append(AuditEntry(**e))
        return log
