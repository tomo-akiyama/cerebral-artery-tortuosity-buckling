#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run-scoped, frame-independent MCA conflict event registry (standard library).

CSV remains the measurement/label interchange format. This SQLite sidecar records
*which import event* created a conflict and *which conflict* a person resolved.
A source snapshot replay is idempotent. A new event id creates a new episode,
even if key, geometry and candidate labels are unchanged. No clinical values or
selection thresholds are stored or changed here.

Default import identity is SHA-256 of the source bytes. Identical bytes supplied
as a genuinely new audit must receive a new --mca-event-id (the same id for both
frames). File names, wall time and current label values are not event identity.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Mapping, Optional

REGISTRY_VERSION = "mca_registry_v1"
REGISTRY_TOKEN_COL = "mca_registry_token"
LABEL_STATE_REVISION = "labels_event_registry_v1"
KEYS = ("directory_id", "segment_id", "segment_index")
GEOMETRY_COL = "centerline_geometry_sha256"


class RegistryStateError(RuntimeError):
    """Missing/contradictory state: stop rather than guess an audit history."""


class StaleLabelViewError(RegistryStateError):
    """The displayed interval belongs to an older conflict episode."""


def text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() in {"", "nan", "none", "null", "<na>"} else s


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def normalized_key(row: Mapping[str, Any]) -> dict[str, str]:
    """1, 1.0 and numpy integer ids identify the same interval; never truncate."""
    out: dict[str, str] = {}
    for k in KEYS:
        v = text(row.get(k))
        if not v:
            raise ValueError(f"Missing interval key: {k}")
        if k != "directory_id":
            try:
                n = Decimal(v)
                if not n.is_finite():
                    raise ValueError(f"Nonfinite interval key: {k}")
                v = format(n.normalize(), "f")
            except InvalidOperation:
                pass
        out[k] = v
    return out


def subject_id(row: Mapping[str, Any]) -> str:
    return digest({"key": normalized_key(row),
                   "geometry": text(row.get(GEOMETRY_COL)).lower()})


def canonical_labels(labels: Any) -> str:
    return "|".join(sorted({v.strip() for v in text(labels).split("|") if v.strip()}))


def registry_path_for_manual(manual_path: Path | str) -> Path:
    return Path(manual_path).parent / "mca_conflict_registry.sqlite3"


def source_identity(path: Path | str, event_id: Optional[str] = None) -> tuple[str, str]:
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if event_id is not None and not text(event_id):
        raise ValueError("mca-event-id must be nonempty")
    return ("explicit:" + text(event_id) if event_id is not None else "sha256:" + sha, sha)


@contextmanager
def label_write_lock(manual_path: Path | str) -> Iterator[None]:
    """Serialize 23/GUI updates to one run on a local filesystem (Mac/Linux/Win).

    The lock is held only during reads/writes, never while the GUI is displayed.
    Lock files are left in place deliberately (unlinking a lock can split locks).
    """
    p = Path(manual_path).parent / ".mca_labels.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a+b") as f:
        if os.name == "nt":
            import msvcrt
            if f.tell() == 0:
                f.write(b"0"); f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class ConflictRegistry:
    """Transactional persistent event journal shared by all frames in a run."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> "ConflictRegistry":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA synchronous=FULL")
        try:
            c.execute("BEGIN IMMEDIATE")
            c.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            row = c.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()
            if row and row[0] != REGISTRY_VERSION:
                raise RegistryStateError(f"Unsupported registry schema: {row[0]}")
            c.execute("INSERT OR IGNORE INTO metadata VALUES ('schema',?)", (REGISTRY_VERSION,))
            c.execute("""CREATE TABLE IF NOT EXISTS sources (
                event_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL)""")
            c.execute("""CREATE TABLE IF NOT EXISTS episodes (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL, event_id TEXT NOT NULL, signature TEXT NOT NULL,
                token TEXT NOT NULL UNIQUE, key_json TEXT NOT NULL, geometry TEXT NOT NULL,
                labels TEXT NOT NULL, conflict_status TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('open','resolved','superseded')),
                resolved_label TEXT NOT NULL DEFAULT '', created_utc TEXT NOT NULL,
                resolved_utc TEXT NOT NULL DEFAULT '',
                UNIQUE(subject,event_id,signature))""")
            c.execute("""CREATE TABLE IF NOT EXISTS heads (
                subject TEXT PRIMARY KEY, token TEXT NOT NULL REFERENCES episodes(token))""")
            c.execute("""CREATE TABLE IF NOT EXISTS aliases (
                subject TEXT NOT NULL, alias TEXT NOT NULL,
                token TEXT NOT NULL REFERENCES episodes(token), PRIMARY KEY(subject,alias))""")
            c.execute("""CREATE TABLE IF NOT EXISTS geometry_bindings (
                unknown_subject TEXT PRIMARY KEY, subject TEXT NOT NULL)""")
            c.execute("""CREATE TABLE IF NOT EXISTS confirmations (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT NOT NULL,
                token TEXT NOT NULL, label TEXT NOT NULL, confirmed_utc TEXT NOT NULL)""")
        except BaseException:
            c.rollback(); c.close(); raise
        self.conn = c
        return self

    def __exit__(self, typ, value, tb) -> None:
        assert self.conn is not None
        try:
            if typ is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            self.conn.close(); self.conn = None

    def _db(self) -> sqlite3.Connection:
        if self.conn is None:
            raise RuntimeError("Use ConflictRegistry as a context manager")
        return self.conn

    def _subject(self, row: Mapping[str, Any]) -> str:
        sub = subject_id(row)
        if not text(row.get(GEOMETRY_COL)):
            r = self._db().execute("SELECT subject FROM geometry_bindings WHERE unknown_subject=?", (sub,)).fetchone()
            if r:
                return str(r[0])
        return sub

    def attach_displayed_geometry(self, old_row: Mapping[str, Any], new_row: Mapping[str, Any]) -> None:
        """Bind hashless legacy state only when a person has displayed actual geometry.

        This does not verify the inherited label. confirm() immediately follows
        and records the person's choice. Competing known geometries are rejected.
        """
        if text(old_row.get(GEOMETRY_COL)) or not text(new_row.get(GEOMETRY_COL)):
            return
        if normalized_key(old_row) != normalized_key(new_row):
            raise ValueError("Cannot bind different interval keys")
        c = self._db(); old = subject_id(old_row); new = subject_id(new_row)
        binding = c.execute("SELECT subject FROM geometry_bindings WHERE unknown_subject=?", (old,)).fetchone()
        if binding:
            if str(binding[0]) != new:
                raise StaleLabelViewError("Hashless legacy state is already bound to a different geometry; rebuild from current measurements")
            return
        old_head = c.execute("SELECT token FROM heads WHERE subject=?", (old,)).fetchone()
        new_head = c.execute("SELECT token FROM heads WHERE subject=?", (new,)).fetchone()
        if old_head and new_head:
            raise StaleLabelViewError("Both hashless and known-geometry conflicts exist; reload a current measured dataset before confirmation")
        if old_head:
            c.execute("UPDATE episodes SET subject=?,geometry=? WHERE subject=?", (new,text(new_row.get(GEOMETRY_COL)).lower(),old))
            c.execute("UPDATE heads SET subject=? WHERE subject=?", (new,old))
            c.execute("UPDATE aliases SET subject=? WHERE subject=?", (new,old))
            c.execute("UPDATE confirmations SET subject=? WHERE subject=?", (new,old))
        c.execute("INSERT INTO geometry_bindings VALUES (?,?)", (old,new))

    def bind_source(self, event_id: str, fingerprint: str) -> None:
        """One explicit event id may not silently be reused for different bytes."""
        if not text(event_id) or not text(fingerprint):
            raise ValueError("event id and fingerprint are required")
        c = self._db()
        old = c.execute("SELECT fingerprint FROM sources WHERE event_id=?", (event_id,)).fetchone()
        if old and old[0] != fingerprint:
            raise RegistryStateError("The same --mca-event-id was used for different source contents; use a new event id")
        c.execute("INSERT OR IGNORE INTO sources VALUES (?,?)", (event_id, fingerprint))

    def current(self, row: Mapping[str, Any]) -> Optional[dict[str, Any]]:
        r = self._db().execute("""SELECT e.* FROM episodes e JOIN heads h ON e.token=h.token
                                   WHERE h.subject=?""", (self._subject(row),)).fetchone()
        return dict(r) if r is not None else None

    def canonical_token(self, row: Mapping[str, Any], token: Any) -> str:
        t = text(token)
        if not t:
            return ""
        sub = self._subject(row)
        c = self._db()
        direct = c.execute("SELECT subject FROM episodes WHERE token=?", (t,)).fetchone()
        if direct:
            if direct[0] != sub:
                raise RegistryStateError("Conflict token belongs to another key/geometry")
            return t
        alias = c.execute("SELECT token FROM aliases WHERE subject=? AND alias=?", (sub,t)).fetchone()
        return str(alias[0]) if alias else ""

    def register(self, row: Mapping[str, Any], labels: Any, status: str, event_id: str,
                 *, legacy_token: str = "") -> dict[str, Any]:
        """Replay returns the current head, never reopens an older event."""
        if not text(event_id):
            raise ValueError("A persistent conflict requires a source event id")
        c = self._db(); sub = self._subject(row)
        labs = canonical_labels(labels)
        signature = digest({"labels": labs, "status": text(status)})
        old = c.execute("SELECT token FROM episodes WHERE subject=? AND event_id=? AND signature=?",
                        (sub,event_id,signature)).fetchone()
        if old:
            return self.current(row)  # type: ignore[return-value]
        head = self.current(row)
        # The new confirmation must cover the union of still unresolved evidence.
        if head and head['state'] == 'open':
            labs = canonical_labels(labs + '|' + head['labels'])
        token = text(legacy_token) or ("mca2:" + digest([sub,event_id,signature]))
        # A legacy token reused for another episode must not overwrite its record.
        occupied = c.execute("SELECT subject FROM episodes WHERE token=?", (token,)).fetchone()
        if occupied:
            token = "mca2:" + digest([sub,event_id,signature])
        c.execute("""INSERT INTO episodes
            (subject,event_id,signature,token,key_json,geometry,labels,conflict_status,state,created_utc)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (sub,event_id,signature,token,json.dumps(normalized_key(row),sort_keys=True),
             text(row.get(GEOMETRY_COL)).lower(),labs,text(status) or 'conflict_unresolved','open',utc_now()))
        if head and head['state'] == 'open':
            c.execute("UPDATE episodes SET state='superseded' WHERE token=?", (head['token'],))
        c.execute("INSERT INTO heads VALUES (?,?) ON CONFLICT(subject) DO UPDATE SET token=excluded.token",
                  (sub,token))
        return self.current(row)  # type: ignore[return-value]

    def observe_existing(self, row: Mapping[str, Any]) -> Optional[dict[str, Any]]:
        """Import a pending legacy episode once; does not turn a retry into a new event."""
        from sin_trend_mca_labels import _row_is_unresolved_conflict
        if not _row_is_unresolved_conflict(row):
            return self.current(row)
        t = text(row.get('mca_conflict_token'))
        canonical = self.canonical_token(row,t)
        if canonical:
            return self.current(row)
        if t.startswith('mca2:'):
            raise RegistryStateError("A registry-issued token is missing from its registry. Restore the registry backup; do not discard it")
        current = self.current(row)
        if current is not None:
            # An unrecognized older CSV is not a new audit event. Keep the
            # current head; do not alias a stale view to it for confirmation.
            return current
        labs = canonical_labels(row.get('mca_conflict_labels')) or text(row.get('mca_segment'))
        status = text(row.get('label_status'))
        if status not in {'conflict_unresolved','shape_mismatch','source_multiple_labels'}:
            status='conflict_unresolved'
        # Legacy migration is a single import, not a newly requested audit.
        event = 'legacy-migration:' + digest([subject_id(row),labs,status])
        state = self.register(row,labs,status,event,legacy_token=t)
        if t:
            self._db().execute("INSERT OR IGNORE INTO aliases VALUES (?,?,?)",
                               (self._subject(row),t,state['token']))
        return state

    def confirm(self, row: Mapping[str, Any], label: str, expected_token: str,
                *, expected_view_token: str = "") -> str:
        """Only a confirmation made against the current episode can close it."""
        if label not in {'M1','M2-3','M4'}:
            raise ValueError(f"Invalid MCA label {label!r}")
        cur=self.current(row); tok=""
        view = self.canonical_token(row,expected_view_token) if text(expected_view_token) else ""
        expected = self.canonical_token(row,expected_token) if text(expected_token) else ""
        if text(expected_token) and not expected:
            raise StaleLabelViewError('Unrecognized displayed conflict; reload from the current shared state')
        if cur:
            tok=cur['token']
            if text(expected_view_token) and view != tok:
                raise StaleLabelViewError('A newer conflict appeared after this view was prepared. Reload and confirm the current interval')
            if cur['state']=='open' and expected != tok:
                raise StaleLabelViewError('The displayed row does not resolve the current conflict. Reload before confirming')
            if expected and expected != tok:
                raise StaleLabelViewError('The confirmation refers to an older conflict')
            self._db().execute("UPDATE episodes SET state='resolved',resolved_label=?,resolved_utc=? WHERE token=?",
                               (label,utc_now(),tok))
        elif text(expected_token) or text(expected_view_token):
            raise RegistryStateError('Unknown confirmation token')
        self._db().execute("INSERT INTO confirmations(subject,token,label,confirmed_utc) VALUES (?,?,?,?)",
                           (self._subject(row),tok,label,utc_now()))
        return tok

    def project_row(self, frame, idx, *, always_resolved: bool = False) -> None:
        """Overlay current conflict state, retaining measurement columns verbatim."""
        from sin_trend_mca_labels import _row_is_unresolved_conflict
        row=frame.loc[idx]; cur=self.current(row)
        if cur is None:
            return
        frame.at[idx,REGISTRY_TOKEN_COL]=cur['token']
        if cur['state']=='open':
            frame.at[idx,'mca_segment']=None
            frame.at[idx,'mca_label_conflict']=True
            frame.at[idx,'label_status']=cur['conflict_status']
            frame.at[idx,'mca_conflict_labels']=cur['labels']
            frame.at[idx,'mca_conflict_token']=cur['token']
            frame.at[idx,'mca_conflict_at_utc']=cur['created_utc']
        elif always_resolved or _row_is_unresolved_conflict(row):
            frame.at[idx,'mca_segment']=cur['resolved_label']
            frame.at[idx,'mca_label_conflict']=False
            frame.at[idx,'label_status']='ok'
            frame.at[idx,'mca_conflict_labels']=''
            frame.at[idx,'mca_conflict_token']=''
            frame.at[idx,'mca_conflict_at_utc']=''
        frame.at[idx,'label_state_revision']=LABEL_STATE_REVISION


def reconcile_registered(analysis, *, base_labels=None, update_labels=None, fill_labels=None,
                         registry: ConflictRegistry, event_id: Optional[str] = None,
                         source_fingerprint: Optional[str] = None):
    """23/GUI bridge: existing geometric label checks plus persistent episodes.

    The numerical measurement pipeline is never called or modified here.
    """
    import pandas as pd
    from sin_trend_mca_labels import (
        ensure_label_status_columns, inherit_mca_labels_layered, _key_tuple,
        _row_is_unresolved_conflict, _active_conflict_report)
    from sin_trend_io_utils import normalize_id_columns

    a=ensure_label_status_columns(normalize_id_columns(analysis))
    b=ensure_label_status_columns(normalize_id_columns(base_labels)) if base_labels is not None else None
    # Adopt saved pending rows before projecting state from another frame.
    for table in (a,b):
        if table is not None:
            for _,r in table.iterrows():
                if _row_is_unresolved_conflict(r):
                    registry.observe_existing(r)
    for i in a.index:
        registry.project_row(a,i)
    if b is not None:
        for i in b.index:
            registry.project_row(b,i)
    if event_id and source_fingerprint:
        registry.bind_source(event_id,source_fingerprint)

    out, _ = inherit_mca_labels_layered(a,base_labels=b,update_labels=update_labels,fill_labels=fill_labels)
    raw_history=out.attrs.get('conflict_history')
    history=raw_history.copy() if isinstance(raw_history,pd.DataFrame) else pd.DataFrame()
    # The existing merge identifies unsafe source rows/hash mismatches. Give
    # each newly observed conflict a stable import identity rather than a label seed.
    if len(history):
        target={_key_tuple(r): i for i,r in out.iterrows()}
        for _,h in history.iterrows():
            if text(h.get('conflict_type')) not in {
                'source_unresolved_conflict', 'source_multiple_labels',
                'source_geometry_hash_mismatch', 'geometry_hash_mismatch',
                'shape_mismatch', 'existing_vs_incoming'}:
                continue
            i=target.get(_key_tuple(h))
            if i is None or not _row_is_unresolved_conflict(out.loc[i]):
                continue
            eid=event_id
            if not eid:
                # fill-only legacy evidence: no new audit is inferred from a retry
                eid='legacy-fill:' + digest([subject_id(out.loc[i]),text(h.get('labels')),text(h.get('conflict_type'))])
            registry.register(out.loc[i],h.get('labels'),text(out.at[i,'label_status']),eid)
    for i in out.index:
        # A new registry conflict cannot be cleared by a pre-existing manual CSV.
        # Only GUI confirm() commits a resolution. Replays of resolved source
        # events project the recorded decision instead of opening it again.
        registry.project_row(out,i)
    active,hist,events=_active_conflict_report(out,[history] if len(history) else [])
    out.attrs['conflict_history']=hist
    out.attrs['conflict_active_events']=events
    out.attrs['n_conflict_history']=len(hist)
    out.attrs['n_mca_conflicts_active']=len(active)
    out.attrs['n_mca_conflict_active_events']=len(events)
    return out,active
