"""Tests for the hash-chained audit log."""
from calibplat.audit import AuditLog


def test_empty_log_verifies():
    log = AuditLog()
    ok, broken = log.verify()
    assert ok and broken is None


def test_append_and_verify():
    log = AuditLog()
    log.append("register", {"version": "v1"})
    log.append("promote", {"version": "v1"})
    log.append("rollback", {"from": "v1", "to": "v0"})
    assert len(log.entries) == 3
    assert log.entries[0].prev_hash == "GENESIS"
    assert log.entries[1].prev_hash == log.entries[0].entry_hash
    ok, broken = log.verify()
    assert ok and broken is None


def test_tampered_payload_breaks_chain():
    log = AuditLog()
    log.append("register", {"version": "v1"})
    log.append("promote", {"version": "v1"})
    log.entries[0].payload["version"] = "vX"
    ok, broken = log.verify()
    assert not ok and broken == 0


def test_deleted_entry_breaks_chain():
    log = AuditLog()
    log.append("register", {"version": "v1"})
    log.append("promote", {"version": "v1"})
    log.append("rollback", {"from": "v1", "to": "v0"})
    del log.entries[1]
    ok, broken = log.verify()
    assert not ok


def test_roundtrip():
    log = AuditLog()
    log.append("register", {"version": "v1"})
    restored = AuditLog.from_dict(log.to_dict())
    ok, _ = restored.verify()
    assert ok
    assert len(restored.entries) == 1
