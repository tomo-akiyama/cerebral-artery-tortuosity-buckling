#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend mca labels."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sin_trend_paths import INTERVAL_KEY, atomic_to_csv, sha256_array, sha256_file, sha256_json
from sin_trend_io_utils import normalize_id_columns

                                           
SHAPE_FINGERPRINT_COLS: Tuple[str, ...] = ("pts", "L_full_mm", "euclid_distance_mm")
GEOMETRY_HASH_COL = "centerline_geometry_sha256"
GEOMETRY_HASH_VERSION = "v1"

LABEL_STATUS_OK = "ok"
LABEL_STATUS_UNSET = "unset"
LABEL_STATUS_INHERITED = "inherited_shape_verified"
LABEL_STATUS_INHERITED_STATS = "inherited_stats_match_unverified"
LABEL_STATUS_INHERITED_ID_ONLY = "inherited_id_only_shape_unverified"
LABEL_STATUS_CONFLICT = "conflict_unresolved"
LABEL_STATUS_SHAPE_MISMATCH = "shape_mismatch"
LABEL_STATUS_SOURCE_MULTI = "source_multiple_labels"

MCA_CONFLICT_TOKEN_COL = "mca_conflict_token"
MCA_CONFLICT_AT_COL = "mca_conflict_at_utc"
RESOLVED_CONFLICT_TOKEN_COL = "resolved_conflict_token"
LAST_RESOLVED_TOKEN_COL = "mca_last_resolved_conflict_token"
LABEL_STATE_REVISION = "labels_event_registry_v1"

_CONFLICT_STATUSES = frozenset(
    {LABEL_STATUS_CONFLICT, LABEL_STATUS_SHAPE_MISMATCH, LABEL_STATUS_SOURCE_MULTI}
)

MANUAL_LABEL_COLUMNS: Tuple[str, ...] = (
    tuple(INTERVAL_KEY)
    + SHAPE_FINGERPRINT_COLS
    + (
        GEOMETRY_HASH_COL,
        "mca_segment",
        "label_status",
        RESOLVED_CONFLICT_TOKEN_COL,
        "labeled_at_utc",
        "label_origin",
    )
)

                                                     
_SHAPE_VERIFIED_STATUSES = frozenset({LABEL_STATUS_OK, LABEL_STATUS_INHERITED})
_GUI_LABEL_ORIGINS = frozenset({"gui", "visual_inspection", "manual", "user"})

VerificationLevel = str  # "geometry" | "stats" | "geometry_mismatch" | "none"


def mca_is_set(val: Any) -> bool:
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except Exception:  # noqa: BLE001
        pass
    s = str(val).strip()
    return bool(s) and s.lower() not in {"nan", "none", "null"}



def _flag_value(value: Any) -> bool:
    """CSV bool parsing: the nonempty string 'False' must not become True."""
    if not mca_is_set(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    ss = str(value).strip().lower()
    if ss in {"false", "0", "0.0", "no", "n", "f"}:
        return False
    if ss in {"true", "1", "1.0", "yes", "y", "t"}:
        return True
    raise ValueError(f"Invalid mca_label_conflict flag: {value!r}")


def ensure_label_status_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "mca_segment": np.nan, "mca_label_conflict": False,
        "label_status": LABEL_STATUS_UNSET, "mca_conflict_labels": "",
        MCA_CONFLICT_TOKEN_COL: "", MCA_CONFLICT_AT_COL: "", GEOMETRY_HASH_COL: "",
        LAST_RESOLVED_TOKEN_COL: "",
    }
    for col, val in defaults.items():
        if col not in out.columns:
            out[col] = pd.Series([val] * len(out), index=out.index, dtype=object)
        elif col != "mca_label_conflict":
            out[col] = out[col].astype(object)
    flag = out["mca_label_conflict"].map(_flag_value)
    status = out["label_status"].map(lambda v: str(v).strip() if mca_is_set(v) else "")
    out["mca_label_conflict"] = flag | status.isin(_CONFLICT_STATUSES)
    return out


def _key_tuple(row: pd.Series, keys: Sequence[str] = INTERVAL_KEY) -> Tuple[Any, ...]:
    return tuple(row[k] for k in keys)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


LEGACY_UNRESOLVED_EPISODE_SEED = "legacy_unresolved_v1"
FRESH_UNRESOLVED_EPISODE_SEED = "fresh_unresolved"


def make_conflict_token(
    key: Dict[str, Any],
    conflict_labels: str,
    *,
    geometry_hash: str = "",
    episode_seed: str = "",
    conflict_at_utc: Optional[str] = None,
) -> str:
    from sin_trend_paths import to_python_scalar as _tps

    _ = conflict_at_utc                            
    payload = {
        "directory_id": key.get("directory_id"),
        "segment_id": key.get("segment_id"),
        "segment_index": key.get("segment_index"),
        "conflict_labels": str(conflict_labels or ""),
        "geometry_hash": str(geometry_hash or "").strip().lower(),
        "episode_seed": str(episode_seed or ""),
    }
    return sha256_json({k: _tps(v) for k, v in payload.items()})


def _row_geometry_hash(row: pd.Series) -> str:
    if GEOMETRY_HASH_COL not in row.index:
        return ""
    return str(row.get(GEOMETRY_HASH_COL) or "").strip()


def episode_seed_for_new_conflict(row: pd.Series) -> str:
    if bool(row.get("mca_label_conflict", False)) or str(row.get("label_status", "") or "").strip() in _CONFLICT_STATUSES:
        if not mca_is_set(row.get(MCA_CONFLICT_TOKEN_COL)):
            return LEGACY_UNRESOLVED_EPISODE_SEED
    # Pure in-memory compatibility path: preserve the resolution lineage.
    # Persistent CLI processing uses sin_trend_mca_registry event identity instead.
    previous = row.get(LAST_RESOLVED_TOKEN_COL, "")
    if mca_is_set(previous):
        return "after-token:" + str(previous).strip()
    lab = row.get("mca_segment", "")
    st = str(row.get("label_status", "") or "").strip()
    if mca_is_set(lab) or st == LABEL_STATUS_OK:
        lab_s = str(lab).strip() if mca_is_set(lab) else ""
        return f"after:{lab_s}:{st or LABEL_STATUS_OK}"
    return FRESH_UNRESOLVED_EPISODE_SEED


def ensure_persistent_conflict_tokens(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_label_status_columns(df)
    for i, row in out.iterrows():
        if not _row_is_unresolved_conflict(row):
            continue
        out.at[i, "mca_label_conflict"] = True
        if mca_is_set(out.at[i, MCA_CONFLICT_TOKEN_COL]):
            continue
        labs = row.get("mca_conflict_labels", "")
        if not mca_is_set(labs) and mca_is_set(row.get("mca_segment")):
            labs = str(row.get("mca_segment")).strip()
        labs_s = str(labs) if mca_is_set(labs) else ""
        if not mca_is_set(out.at[i, "mca_conflict_labels"]) and labs_s:
            out.at[i, "mca_conflict_labels"] = labs_s
        key = {k: out.at[i, k] for k in INTERVAL_KEY}
        tok = make_conflict_token(
            key,
            labs_s,
            geometry_hash=_row_geometry_hash(out.loc[i]),
            episode_seed=LEGACY_UNRESOLVED_EPISODE_SEED,
        )
        out.at[i, MCA_CONFLICT_TOKEN_COL] = tok
        if not mca_is_set(out.at[i, MCA_CONFLICT_AT_COL]):
            out.at[i, MCA_CONFLICT_AT_COL] = _utc_now_iso()
    return out


def _row_is_unresolved_conflict(row: pd.Series) -> bool:
    if _flag_value(row.get("mca_label_conflict", False)):
        return True
    st = str(row.get("label_status", "") or "").strip()
    return st in _CONFLICT_STATUSES


def _mark_row_conflict(
    out: pd.DataFrame,
    i: Any,
    conflict_labels: str,
    *,
    status: str = LABEL_STATUS_CONFLICT,
    token: Optional[str] = None,
    conflict_at_utc: Optional[str] = None,
    episode_seed: Optional[str] = None,
) -> str:
    labs = str(conflict_labels or "")
    at = conflict_at_utc or ""
    existing_token = out.at[i, MCA_CONFLICT_TOKEN_COL] if MCA_CONFLICT_TOKEN_COL in out.columns else ""
    was_conflict = bool(out.at[i, "mca_label_conflict"])
    if mca_is_set(existing_token) and was_conflict:
        tok = str(existing_token).strip()
        if not mca_is_set(at):
            at = str(out.at[i, MCA_CONFLICT_AT_COL] or "") if MCA_CONFLICT_AT_COL in out.columns else ""
    else:
        if not mca_is_set(at):
            at = _utc_now_iso()
        if mca_is_set(token):
            tok = str(token).strip()
        else:
            key = {k: out.at[i, k] for k in INTERVAL_KEY}
            seed = episode_seed if episode_seed is not None else episode_seed_for_new_conflict(out.loc[i])
            tok = make_conflict_token(
                key,
                labs,
                geometry_hash=_row_geometry_hash(out.loc[i]),
                episode_seed=str(seed),
                conflict_at_utc=at,
            )
    out.at[i, "mca_segment"] = np.nan
    out.at[i, "mca_label_conflict"] = True
    out.at[i, "label_status"] = status
    out.at[i, "mca_conflict_labels"] = labs
    out.at[i, MCA_CONFLICT_TOKEN_COL] = tok
    out.at[i, MCA_CONFLICT_AT_COL] = at
    return tok


def _clear_row_conflict(out: pd.DataFrame, i: Any) -> None:
    prior_token = out.at[i, MCA_CONFLICT_TOKEN_COL]
    if mca_is_set(prior_token):
        out.at[i, LAST_RESOLVED_TOKEN_COL] = str(prior_token).strip()
    out.at[i, "mca_label_conflict"] = False
    out.at[i, "mca_conflict_labels"] = ""
    out.at[i, MCA_CONFLICT_TOKEN_COL] = ""
    out.at[i, MCA_CONFLICT_AT_COL] = ""


def compute_centerline_geometry_hash(
    P_vox: np.ndarray,
    *,
    voxel_size_mm: float,
    coordinate_frame: str = "vox",
    coordinate_unit: str = "voxel",
) -> str:
    P = np.asarray(P_vox, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"P_vox must be (N,3), got {P.shape}")
    payload = {
        "version": GEOMETRY_HASH_VERSION,
        "coordinate_frame": str(coordinate_frame),
        "coordinate_unit": str(coordinate_unit),
        "voxel_size_mm": float(voxel_size_mm),
        "n_points": int(P.shape[0]),
        "points_sha256": sha256_array(P),
    }
    return sha256_json(payload)


def shape_fingerprint_available(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in SHAPE_FINGERPRINT_COLS)


def geometry_hash_available_row(row: pd.Series) -> bool:
    return GEOMETRY_HASH_COL in row.index and mca_is_set(row.get(GEOMETRY_HASH_COL))


def geometry_hashes_match(a: pd.Series, b: pd.Series) -> bool:
    if not geometry_hash_available_row(a) or not geometry_hash_available_row(b):
        return False
    return str(a[GEOMETRY_HASH_COL]).strip().lower() == str(b[GEOMETRY_HASH_COL]).strip().lower()


def shape_fingerprints_match(
    a: pd.Series,
    b: pd.Series,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> bool:
    for c in SHAPE_FINGERPRINT_COLS:
        if c not in a.index or c not in b.index:
            return False
        va = a[c]
        vb = b[c]
        try:
            if pd.isna(va) or pd.isna(vb):
                return False
        except Exception:  # noqa: BLE001
            return False
        if c == "pts":
            try:
                if int(va) != int(vb):
                    return False
            except Exception:  # noqa: BLE001
                return False
            continue
        try:
            fa = float(va)
            fb = float(vb)
        except Exception:  # noqa: BLE001
            return False
        if not (np.isfinite(fa) and np.isfinite(fb)):
            return False
        if not np.isclose(fa, fb, rtol=rtol, atol=atol):
            return False
    return True


def shape_verification_level(a: pd.Series, b: pd.Series) -> VerificationLevel:
    a_has = geometry_hash_available_row(a)
    b_has = geometry_hash_available_row(b)
    if a_has and b_has:
        if geometry_hashes_match(a, b):
            return "geometry"
        return "geometry_mismatch"
    if all(c in a.index and c in b.index for c in SHAPE_FINGERPRINT_COLS) and shape_fingerprints_match(a, b):
        return "stats"
    return "none"


def _status_for_verification(level: VerificationLevel) -> str:
    if level == "geometry":
        return LABEL_STATUS_INHERITED
    if level == "stats":
        return LABEL_STATUS_INHERITED_STATS
    return LABEL_STATUS_INHERITED_ID_ONLY


def _prior_status_from_label_row(incoming: pd.Series) -> str:
    if "label_status" in incoming.index and mca_is_set(incoming.get("label_status")):
        return str(incoming.get("label_status")).strip()
    origin = str(incoming.get("label_origin", "") or "").strip().lower()
    if origin in _GUI_LABEL_ORIGINS:
        return LABEL_STATUS_OK
    return LABEL_STATUS_INHERITED_ID_ONLY


def _is_gui_confirmed_label(incoming: pd.Series) -> bool:
    status = ""
    if "label_status" in incoming.index and mca_is_set(incoming.get("label_status")):
        status = str(incoming.get("label_status")).strip()
    if status == LABEL_STATUS_OK:
        return True
    origin = str(incoming.get("label_origin", "") or "").strip().lower()
    return origin in _GUI_LABEL_ORIGINS


def _label_may_resolve_conflicts(incoming: pd.Series, current_row: pd.Series) -> bool:
    if not _is_gui_confirmed_label(incoming):
        return False
    cur_token = current_row.get(MCA_CONFLICT_TOKEN_COL, "") if MCA_CONFLICT_TOKEN_COL in current_row.index else ""
    res_token = (
        incoming.get(RESOLVED_CONFLICT_TOKEN_COL, "")
        if RESOLVED_CONFLICT_TOKEN_COL in incoming.index
        else ""
    )
    if not mca_is_set(cur_token) or not mca_is_set(res_token):
        return False
    return str(cur_token).strip() == str(res_token).strip()


def _finalize_inherited_status(prior_status: str, level: VerificationLevel) -> str:
    from_level = _status_for_verification(level) if level != "none" else LABEL_STATUS_INHERITED_ID_ONLY
    prior = str(prior_status).strip() if prior_status else LABEL_STATUS_INHERITED_ID_ONLY

    if prior == LABEL_STATUS_OK:
        return LABEL_STATUS_OK

    if from_level == LABEL_STATUS_INHERITED and prior not in _SHAPE_VERIFIED_STATUSES:
                                                  
        if prior == LABEL_STATUS_INHERITED_STATS:
            return LABEL_STATUS_INHERITED_STATS
        return LABEL_STATUS_INHERITED_ID_ONLY

    return from_level


def unique_labels_by_key(work: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keys = list(INTERVAL_KEY)
    work = ensure_label_status_columns(normalize_id_columns(work))
    cols = keys + ["mca_segment", "mca_label_conflict", "mca_conflict_labels", "label_status"]
    for c in SHAPE_FINGERPRINT_COLS:
        if c in work.columns:
            cols.append(c)
    if GEOMETRY_HASH_COL in work.columns:
        cols.append(GEOMETRY_HASH_COL)
    for c in ("label_origin", RESOLVED_CONFLICT_TOKEN_COL, MCA_CONFLICT_TOKEN_COL, MCA_CONFLICT_AT_COL, LAST_RESOLVED_TOKEN_COL):
        if c in work.columns and c not in cols:
            cols.append(c)

    conflicts: List[dict] = []
    safe: List[dict] = []
    seen_keys: set = set()

                            
    for _, r in work.iterrows():
        kt = _key_tuple(r, keys)
        if kt in seen_keys:
            continue
        if not _row_is_unresolved_conflict(r):
            continue
        seen_keys.add(kt)
        clab = r.get("mca_conflict_labels", "")
        if not mca_is_set(clab) and mca_is_set(r.get("mca_segment")):
            clab = str(r.get("mca_segment")).strip()
        conflicts.append(
            {
                **{k: r[k] for k in keys},
                "labels": str(clab) if mca_is_set(clab) else "",
                "conflict_type": "source_unresolved_conflict",
                MCA_CONFLICT_TOKEN_COL: r.get(MCA_CONFLICT_TOKEN_COL, ""),
                MCA_CONFLICT_AT_COL: r.get(MCA_CONFLICT_AT_COL, ""),
            }
        )

    labeled = work.loc[work["mca_segment"].map(mca_is_set), cols].copy()
    if len(labeled) == 0:
        return pd.DataFrame(columns=cols), pd.DataFrame(conflicts)

    for ids, g in labeled.groupby(keys, dropna=False):
        kd = dict(zip(keys, ids if isinstance(ids, tuple) else (ids,)))
        kt = tuple(kd[k] for k in keys)
        if kt in seen_keys:
            continue
                                  
        if g.apply(_row_is_unresolved_conflict, axis=1).any():
            clabs = []
            for _, rr in g.iterrows():
                if mca_is_set(rr.get("mca_conflict_labels")):
                    clabs.append(str(rr.get("mca_conflict_labels")).strip())
                elif mca_is_set(rr.get("mca_segment")):
                    clabs.append(str(rr.get("mca_segment")).strip())
            conflicts.append(
                {
                    **kd,
                    "labels": "|".join(sorted(set(clabs))) if clabs else "",
                    "conflict_type": "source_unresolved_conflict",
                }
            )
            seen_keys.add(kt)
            continue

        labs = sorted({str(x).strip() for x in g["mca_segment"] if mca_is_set(x)})
        if len(labs) > 1:
            conflicts.append({**kd, "labels": "|".join(labs), "conflict_type": "source_multiple_labels"})
            seen_keys.add(kt)
            continue
        if len(labs) != 1:
            continue
        if GEOMETRY_HASH_COL in g.columns:
            hashes = sorted(
                {str(x).strip().lower() for x in g[GEOMETRY_HASH_COL].tolist() if mca_is_set(x)}
            )
            if len(hashes) > 1:
                conflicts.append(
                    {
                        **kd,
                        "labels": labs[0],
                        "conflict_type": "source_geometry_hash_mismatch",
                        "geometry_hashes": "|".join(hashes),
                    }
                )
                seen_keys.add(kt)
                continue
        row = {**kd, "mca_segment": labs[0]}
        for c in SHAPE_FINGERPRINT_COLS:
            if c in g.columns:
                row[c] = g.iloc[0][c]
        if GEOMETRY_HASH_COL in g.columns:
            present = [x for x in g[GEOMETRY_HASH_COL].tolist() if mca_is_set(x)]
            row[GEOMETRY_HASH_COL] = present[0] if present else g.iloc[0][GEOMETRY_HASH_COL]
        if "label_status" in g.columns:
            statuses = [str(x).strip() for x in g["label_status"].tolist() if mca_is_set(x)]
            if LABEL_STATUS_OK in statuses:
                row["label_status"] = LABEL_STATUS_OK
            elif LABEL_STATUS_INHERITED in statuses:
                row["label_status"] = LABEL_STATUS_INHERITED
            elif statuses:
                row["label_status"] = statuses[0]
        if "label_origin" in g.columns:
            origins = [str(x).strip() for x in g["label_origin"].tolist() if mca_is_set(x)]
            if origins:
                row["label_origin"] = origins[0]
        if RESOLVED_CONFLICT_TOKEN_COL in g.columns:
            toks = [str(x).strip() for x in g[RESOLVED_CONFLICT_TOKEN_COL].tolist() if mca_is_set(x)]
            if toks:
                row[RESOLVED_CONFLICT_TOKEN_COL] = toks[0]
        if LAST_RESOLVED_TOKEN_COL in g.columns:
            values = [str(v).strip() for v in g[LAST_RESOLVED_TOKEN_COL] if mca_is_set(v)]
            if values:
                row[LAST_RESOLVED_TOKEN_COL] = values[0]
        safe.append(row)
        seen_keys.add(kt)
    return pd.DataFrame(safe), pd.DataFrame(conflicts)


def apply_manual_labels(
    analysis: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    require_shape: bool = True,
    allow_resolve_conflicts: bool = False,
    authoritative: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = ensure_label_status_columns(normalize_id_columns(analysis))
    keys = list(INTERVAL_KEY)
    conflict_rows: List[dict] = []

    if labels is None or len(labels) == 0:
        unset = ~out["mca_segment"].map(mca_is_set)
        out.loc[unset & ~out["mca_label_conflict"], "label_status"] = LABEL_STATUS_UNSET
        return out, pd.DataFrame(columns=keys + ["labels", "conflict_type"])

    lab = normalize_id_columns(labels)
    if "mca_segment" not in lab.columns and "mca_label_conflict" not in lab.columns:
        raise ValueError("labels table missing mca_segment")
    if "mca_segment" not in lab.columns:
        lab["mca_segment"] = np.nan
    safe, multi = unique_labels_by_key(lab)
    if len(multi):
        conflict_rows.extend(multi.to_dict(orient="records"))

                                                              
    if len(multi):
        multi_map = {_key_tuple(r, keys): r for _, r in multi.iterrows()}
        for i, row in out.iterrows():
            kt = _key_tuple(row, keys)
            if kt not in multi_map:
                continue
            src = multi_map[kt]
            labs = str(src.get("labels", "") or "")
            ctype = str(src.get("conflict_type", "") or "")
            status = LABEL_STATUS_SOURCE_MULTI if ctype == "source_multiple_labels" else LABEL_STATUS_CONFLICT
            tok = src.get(MCA_CONFLICT_TOKEN_COL, "") if MCA_CONFLICT_TOKEN_COL in src.index else ""
            at = src.get(MCA_CONFLICT_AT_COL, "") if MCA_CONFLICT_AT_COL in src.index else ""
            seed: Optional[str] = None
            token_arg: Optional[str] = str(tok).strip() if mca_is_set(tok) else None
            if token_arg is None and ctype == "source_unresolved_conflict":
                                                           
                seed = LEGACY_UNRESOLVED_EPISODE_SEED
            _mark_row_conflict(
                out,
                i,
                labs,
                status=status,
                token=token_arg,
                conflict_at_utc=str(at) if mca_is_set(at) else None,
                episode_seed=seed,
            )

    if len(safe) == 0:
        return out, pd.DataFrame(conflict_rows)

    smap: Dict[Tuple[Any, ...], pd.Series] = {}
    for _, r in safe.iterrows():
        smap[_key_tuple(r, keys)] = r

    for i, row in out.iterrows():
        kt = _key_tuple(row, keys)
        if kt not in smap:
            if not bool(out.at[i, "mca_label_conflict"]) and not mca_is_set(out.at[i, "mca_segment"]):
                out.at[i, "label_status"] = LABEL_STATUS_UNSET
            continue

        incoming = smap[kt]
                           
        cur = out.loc[i]
        was_conflict = bool(cur.get("mca_label_conflict", False))
        if was_conflict:
            if authoritative:
                if not _label_may_resolve_conflicts(incoming, cur):
                    continue
            elif not allow_resolve_conflicts:
                continue

        incoming_label = str(incoming["mca_segment"]).strip()
        if (not mca_is_set(out.at[i, LAST_RESOLVED_TOKEN_COL])
                and mca_is_set(incoming.get(LAST_RESOLVED_TOKEN_COL))):
            out.at[i, LAST_RESOLVED_TOKEN_COL] = str(incoming[LAST_RESOLVED_TOKEN_COL]).strip()
        level = shape_verification_level(out.loc[i], incoming)

        if level == "geometry_mismatch":
            conflict_rows.append(
                {
                    **{k: row[k] for k in keys},
                    "labels": incoming_label,
                    "conflict_type": "geometry_hash_mismatch",
                    "existing": row.get("mca_segment"),
                    "incoming": incoming_label,
                }
            )
            _mark_row_conflict(out, i, incoming_label, status=LABEL_STATUS_SHAPE_MISMATCH)
            continue

        if level == "none":
            has_any_shape_analysis = geometry_hash_available_row(out.loc[i]) or all(
                c in out.loc[i].index and not (pd.isna(out.loc[i][c]) if c in out.loc[i].index else True)
                for c in SHAPE_FINGERPRINT_COLS
            )
            has_any_shape_label = geometry_hash_available_row(incoming) or all(
                c in incoming.index and not (pd.isna(incoming[c]) if c in incoming.index else True)
                for c in SHAPE_FINGERPRINT_COLS
            )
            if require_shape:
                if has_any_shape_analysis and has_any_shape_label:
                    conflict_rows.append(
                        {
                            **{k: row[k] for k in keys},
                            "labels": incoming_label,
                            "conflict_type": "shape_mismatch",
                            "existing": row.get("mca_segment"),
                            "incoming": incoming_label,
                        }
                    )
                    _mark_row_conflict(out, i, incoming_label, status=LABEL_STATUS_SHAPE_MISMATCH)
                    continue
                if has_any_shape_analysis and not has_any_shape_label:
                    conflict_rows.append(
                        {
                            **{k: row[k] for k in keys},
                            "labels": incoming_label,
                            "conflict_type": "missing_shape_fingerprint_on_label",
                            "incoming": incoming_label,
                        }
                    )
                    continue
                continue
            level_for_status: VerificationLevel = "none"
        else:
            level_for_status = level

        prior_status = _prior_status_from_label_row(incoming)
        status_ok = _finalize_inherited_status(prior_status, level_for_status)

        existing_set = mca_is_set(out.at[i, "mca_segment"])

        if authoritative:
            out.at[i, "mca_segment"] = incoming_label
            out.at[i, "label_status"] = status_ok
            _clear_row_conflict(out, i)
            continue

        if existing_set and not was_conflict:
            existing = str(out.at[i, "mca_segment"]).strip()
            if existing != incoming_label:
                labs = f"{existing}|{incoming_label}"
                conflict_rows.append(
                    {
                        **{k: row[k] for k in keys},
                        "labels": labs,
                        "conflict_type": "existing_vs_incoming",
                        "existing": existing,
                        "incoming": incoming_label,
                    }
                )
                _mark_row_conflict(out, i, labs, status=LABEL_STATUS_CONFLICT)
            else:
                existing_prior = str(out.at[i, "label_status"]).strip() if mca_is_set(out.at[i, "label_status"]) else prior_status
                out.at[i, "label_status"] = _finalize_inherited_status(existing_prior, level_for_status)
                _clear_row_conflict(out, i)
        else:
            out.at[i, "mca_segment"] = incoming_label
            out.at[i, "label_status"] = status_ok
            _clear_row_conflict(out, i)

    return out, pd.DataFrame(conflict_rows)


def source_content_fingerprint(source_csv: Path) -> str:
    return sha256_file(source_csv)


def manual_labels_fingerprint(path: Path) -> str:
    if path is None or not Path(path).is_file():
        return ""
    return sha256_file(path)


def should_refresh_work(
    *,
    work_exists: bool,
    work_source_sha: Optional[str],
    current_source_sha: str,
    work_revision: str,
    source_revision: str,
    work_manual_sha: Optional[str] = None,
    current_manual_sha: Optional[str] = None,
) -> Tuple[bool, str]:
    if not work_exists:
        return True, "work_missing"
    if work_source_sha and work_source_sha != current_source_sha:
        return True, "source_content_changed"
    if not work_source_sha:
        return True, "missing_source_sha_metadata"
    if work_revision and source_revision and work_revision != source_revision:
        return True, "analysis_revision_changed"

    cur_m = (current_manual_sha or "").strip()
    wrk_m = (work_manual_sha or "").strip()
    if cur_m != wrk_m:
        if not wrk_m and cur_m:
            return True, "missing_manual_labels_sha_metadata"
        if wrk_m and not cur_m:
            return True, "manual_labels_removed"
        return True, "manual_labels_changed"
    return False, "reuse"


def _confirmed_label_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0 or "mca_segment" not in df.columns:
        return pd.DataFrame(columns=list(INTERVAL_KEY) + ["mca_segment", "label_status"])
    prev = ensure_label_status_columns(normalize_id_columns(df))
    ok = prev["mca_segment"].map(mca_is_set) & ~prev["mca_label_conflict"].fillna(False).astype(bool)
    if not ok.any():
        return pd.DataFrame(columns=list(INTERVAL_KEY) + ["mca_segment", "label_status"])
    use_cols = list(INTERVAL_KEY) + ["mca_segment", "label_status", LAST_RESOLVED_TOKEN_COL]
    for c in SHAPE_FINGERPRINT_COLS:
        if c in prev.columns:
            use_cols.append(c)
    if GEOMETRY_HASH_COL in prev.columns:
        use_cols.append(GEOMETRY_HASH_COL)
    return prev.loc[ok, use_cols].copy()


def _unresolved_conflict_table(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    cols = list(INTERVAL_KEY) + [
        "mca_segment",
        "mca_label_conflict",
        "label_status",
        "mca_conflict_labels",
        MCA_CONFLICT_TOKEN_COL,
        MCA_CONFLICT_AT_COL,
    ]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)
    prev = ensure_label_status_columns(normalize_id_columns(df))
    if "mca_label_conflict" not in prev.columns:
        return pd.DataFrame(columns=cols)
    ok = prev["mca_label_conflict"].fillna(False).astype(bool)
    if not ok.any():
        return pd.DataFrame(columns=cols)
    use = [c for c in cols if c in prev.columns]
    return prev.loc[ok, use].copy()


def _carry_unresolved_conflicts(
    analysis: pd.DataFrame,
    conflict_src: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = ensure_label_status_columns(normalize_id_columns(analysis))
    tab = _unresolved_conflict_table(conflict_src)
    keys = list(INTERVAL_KEY)
    carried_rows: List[dict] = []
    if len(tab) == 0:
        return out, pd.DataFrame(columns=keys + ["labels", "conflict_type"])

    cmap: Dict[Tuple[Any, ...], pd.Series] = {}
    for _, r in tab.iterrows():
        cmap[_key_tuple(r, keys)] = r

    for i, row in out.iterrows():
        kt = _key_tuple(row, keys)
        if kt not in cmap:
            continue
        src = cmap[kt]
        clab = src.get("mca_conflict_labels", "")
        if not mca_is_set(clab):
            clab = ""
        status = src.get("label_status", LABEL_STATUS_CONFLICT)
        if not mca_is_set(status):
            status = LABEL_STATUS_CONFLICT
        tok = src.get(MCA_CONFLICT_TOKEN_COL, "") if MCA_CONFLICT_TOKEN_COL in src.index else ""
        at = src.get(MCA_CONFLICT_AT_COL, "") if MCA_CONFLICT_AT_COL in src.index else ""
        token_arg: Optional[str] = str(tok).strip() if mca_is_set(tok) else None
        seed: Optional[str] = None
        if token_arg is None:
                                                            
            seed = LEGACY_UNRESOLVED_EPISODE_SEED
            key = {k: out.at[i, k] for k in keys}
            token_arg = make_conflict_token(
                key,
                str(clab),
                geometry_hash=_row_geometry_hash(out.loc[i]) or _row_geometry_hash(src),
                episode_seed=seed,
            )
        _mark_row_conflict(
            out,
            i,
            str(clab),
            status=str(status).strip(),
            token=token_arg,
            conflict_at_utc=str(at) if mca_is_set(at) else None,
            episode_seed=seed,
        )
        carried_rows.append(
            {
                **{k: row[k] for k in keys},
                "labels": str(clab),
                "conflict_type": "carried_unresolved",
                MCA_CONFLICT_TOKEN_COL: out.at[i, MCA_CONFLICT_TOKEN_COL],
                "existing": out.at[i, "mca_segment"],
                "incoming": "",
            }
        )
    return out, pd.DataFrame(carried_rows)


def _active_conflict_report(
    out: pd.DataFrame,
    conflict_parts: List[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = list(INTERVAL_KEY)
    empty_cols = keys + ["labels", "conflict_type"]
    if conflict_parts:
        history = pd.concat(conflict_parts, ignore_index=True, sort=False)
    else:
        history = pd.DataFrame(columns=empty_cols)

    conflict_mask = out["mca_label_conflict"].fillna(False).astype(bool)
    still_rows = out.loc[conflict_mask]
    still = {_key_tuple(r, keys) for _, r in still_rows.iterrows()}
    if not still:
        empty = pd.DataFrame(columns=empty_cols)
        return empty, history, empty

                          
    if len(history):
        keep = history.apply(lambda r: _key_tuple(r, keys) in still, axis=1)
        active_events = history.loc[keep].copy()
    else:
        active_events = pd.DataFrame(columns=empty_cols)
    covered = {_key_tuple(r, keys) for _, r in active_events.iterrows()} if len(active_events) else set()
    for _, r in still_rows.iterrows():
        kt = _key_tuple(r, keys)
        if kt in covered:
            continue
        active_events = pd.concat(
            [
                active_events,
                pd.DataFrame(
                    [
                        {
                            **{k: r[k] for k in keys},
                            "labels": str(r.get("mca_conflict_labels", "") or ""),
                            "conflict_type": "unresolved_active",
                            MCA_CONFLICT_TOKEN_COL: r.get(MCA_CONFLICT_TOKEN_COL, ""),
                            MCA_CONFLICT_AT_COL: r.get(MCA_CONFLICT_AT_COL, ""),
                        }
                    ]
                ),
            ],
            ignore_index=True,
            sort=False,
        )
        covered.add(kt)

                               
    interval_rows: List[dict] = []
    for _, r in still_rows.iterrows():
        kt = _key_tuple(r, keys)
        ctype = "unresolved_active"
        labs = str(r.get("mca_conflict_labels", "") or "")
        if len(active_events):
            ev_match = active_events.apply(lambda er: _key_tuple(er, keys) == kt, axis=1)
            if ev_match.any():
                ev0 = active_events.loc[ev_match].iloc[0]
                if mca_is_set(ev0.get("conflict_type")):
                    ctype = str(ev0.get("conflict_type"))
                if mca_is_set(ev0.get("labels")):
                    labs = str(ev0.get("labels"))
        interval_rows.append(
            {
                **{k: r[k] for k in keys},
                "labels": labs,
                "conflict_type": ctype,
                MCA_CONFLICT_TOKEN_COL: r.get(MCA_CONFLICT_TOKEN_COL, ""),
                MCA_CONFLICT_AT_COL: r.get(MCA_CONFLICT_AT_COL, ""),
            }
        )
    active_intervals = pd.DataFrame(interval_rows) if interval_rows else pd.DataFrame(columns=empty_cols)
    return active_intervals, history, active_events


def build_work_from_source(
    source_df: pd.DataFrame,
    *,
    previous_work: Optional[pd.DataFrame] = None,
    manual_labels: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = ensure_persistent_conflict_tokens(normalize_id_columns(source_df))
    conflict_parts: List[pd.DataFrame] = []

                                         
    if manual_labels is not None and len(manual_labels):
        base, conf = apply_manual_labels(
            base,
            normalize_id_columns(manual_labels),
            require_shape=True,
            allow_resolve_conflicts=True,
            authoritative=True,
        )
        if len(conf):
            conflict_parts.append(conf)

                                                   
    prev_labels = _confirmed_label_table(previous_work) if previous_work is not None else pd.DataFrame()
    if len(prev_labels):
        keys = list(INTERVAL_KEY)
        unset_nonconflict = (~base["mca_segment"].map(mca_is_set)) & ~base["mca_label_conflict"].fillna(False).astype(bool)
        allowed_keys = {_key_tuple(r, keys) for _, r in base.loc[unset_nonconflict].iterrows()}
        if allowed_keys:
            keep = prev_labels.apply(lambda r: _key_tuple(r, keys) in allowed_keys, axis=1)
            prev_subset = prev_labels.loc[keep]
            if len(prev_subset):
                base, conf = apply_manual_labels(
                    base,
                    prev_subset,
                    require_shape=True,
                    allow_resolve_conflicts=False,
                    authoritative=False,
                )
                if len(conf):
                    conflict_parts.append(conf)

    if conflict_parts:
        conflicts = pd.concat(conflict_parts, ignore_index=True, sort=False)
    else:
        conflicts = pd.DataFrame(columns=list(INTERVAL_KEY) + ["labels", "conflict_type"])
    return base, conflicts


def sync_confirmed_labels_to_manual(
    df: pd.DataFrame,
    path: Path,
    *,
    origin: str = "sync_from_work",
    fill_only: bool = True,
) -> Tuple[pd.DataFrame, int]:
    confirmed = _confirmed_label_table(df)
    labels = load_manual_labels(path)
    existing: set = set()
    if len(labels) and "mca_segment" in labels.columns:
        for _, r in labels.iterrows():
            if mca_is_set(r.get("mca_segment")):
                existing.add(_key_tuple(r, INTERVAL_KEY))
    n_added = 0
    if len(confirmed) == 0:
        return labels, 0
    for _, row in confirmed.iterrows():
        kt = _key_tuple(row, INTERVAL_KEY)
        if fill_only and kt in existing:
            continue
        key = {k: row[k] for k in INTERVAL_KEY}
        shape = {c: row[c] for c in SHAPE_FINGERPRINT_COLS if c in row.index}
        if GEOMETRY_HASH_COL in row.index and mca_is_set(row.get(GEOMETRY_HASH_COL)):
            shape[GEOMETRY_HASH_COL] = row[GEOMETRY_HASH_COL]
        row_status = str(row["label_status"]).strip() if "label_status" in row.index and mca_is_set(row.get("label_status")) else LABEL_STATUS_INHERITED_ID_ONLY
        labels = upsert_manual_label_row(
            labels,
            key,
            str(row["mca_segment"]).strip(),
            shape,
            origin=origin,
            label_status=row_status,
        )
        existing.add(kt)
        n_added += 1
    if n_added:
        save_manual_labels(labels, path)
    return labels, n_added


def inherit_mca_labels_layered(
    analysis: pd.DataFrame,
    *,
    base_labels: Optional[pd.DataFrame] = None,
    update_labels: Optional[pd.DataFrame] = None,
    fill_labels: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = ensure_persistent_conflict_tokens(normalize_id_columns(analysis))
    conflict_parts: List[pd.DataFrame] = []
    base_ensured = (
        ensure_persistent_conflict_tokens(normalize_id_columns(base_labels))
        if base_labels is not None
        else None
    )

                                                
    if base_ensured is not None:
        out, carried = _carry_unresolved_conflicts(out, base_ensured)
        if len(carried):
            conflict_parts.append(carried)

                                           
    base_tab = _confirmed_label_table(base_ensured) if base_ensured is not None else pd.DataFrame()
    if len(base_tab):
        req = shape_fingerprint_available(base_tab) and shape_fingerprint_available(out)
        out, conf = apply_manual_labels(
            out,
            base_tab,
            require_shape=req,
            allow_resolve_conflicts=False,
            authoritative=False,
        )
        if len(conf):
            conflict_parts.append(conf)

                                             
    if update_labels is not None and len(update_labels):
        upd = normalize_id_columns(update_labels)
        req = shape_fingerprint_available(upd)
        out, conf = apply_manual_labels(
            out,
            upd,
            require_shape=req,
            allow_resolve_conflicts=True,
            authoritative=True,
        )
        if len(conf):
            conflict_parts.append(conf)

                                                  
    if fill_labels is not None and len(fill_labels) and "mca_segment" in fill_labels.columns:
        fill = normalize_id_columns(fill_labels)
        keys = list(INTERVAL_KEY)
        unset_nonconflict = (~out["mca_segment"].map(mca_is_set)) & ~out["mca_label_conflict"].fillna(False).astype(bool)
        allowed = {_key_tuple(r, keys) for _, r in out.loc[unset_nonconflict].iterrows()}
        if allowed:
            keep = fill.apply(lambda r: _key_tuple(r, keys) in allowed, axis=1)
            fill_subset = fill.loc[keep]
            if len(fill_subset):
                req = shape_fingerprint_available(fill_subset)
                out, conf = apply_manual_labels(
                    out,
                    fill_subset,
                    require_shape=req,
                    allow_resolve_conflicts=False,
                    authoritative=False,
                )
                if len(conf):
                    conflict_parts.append(conf)

    active, history, active_events = _active_conflict_report(out, conflict_parts)
    out.attrs["conflict_history"] = history
    out.attrs["conflict_active_events"] = active_events
    out.attrs["n_conflict_history"] = int(len(history))
    out.attrs["n_mca_conflicts_active"] = int(len(active))  # unique unresolved intervals
    out.attrs["n_mca_conflict_active_events"] = int(len(active_events))
    return out, active


def upsert_manual_label_row(
    labels_df: pd.DataFrame,
    key: Dict[str, Any],
    label: str,
    shape: Dict[str, Any],
    *,
    origin: str = "gui",
    label_status: Optional[str] = None,
    resolved_conflict_token: Optional[str] = None,
) -> pd.DataFrame:
    keys = list(INTERVAL_KEY)
    out = labels_df.copy() if labels_df is not None and len(labels_df) else pd.DataFrame(columns=list(MANUAL_LABEL_COLUMNS))
    for c in MANUAL_LABEL_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    out = out.astype(object).reset_index(drop=True)
    mask = pd.Series(True, index=out.index) if len(out) else pd.Series(dtype=bool)
    if len(out):
        for k in keys:
            mask &= out[k] == key[k]
    if label_status is None:
        status = LABEL_STATUS_OK if str(origin).strip().lower() in _GUI_LABEL_ORIGINS else LABEL_STATUS_INHERITED_ID_ONLY
    else:
        status = str(label_status).strip() if mca_is_set(label_status) else LABEL_STATUS_INHERITED_ID_ONLY
    new_geom = str(shape.get(GEOMETRY_HASH_COL, "") or "").strip()
    res_tok = ""
    if resolved_conflict_token is not None and mca_is_set(resolved_conflict_token):
        res_tok = str(resolved_conflict_token).strip()
    elif RESOLVED_CONFLICT_TOKEN_COL in shape and mca_is_set(shape.get(RESOLVED_CONFLICT_TOKEN_COL)):
        res_tok = str(shape.get(RESOLVED_CONFLICT_TOKEN_COL)).strip()
    elif len(out) and mask.any() and RESOLVED_CONFLICT_TOKEN_COL in out.columns:
                                                        
        prev_tok = out.loc[mask, RESOLVED_CONFLICT_TOKEN_COL].iloc[0]
        prev_geom = ""
        if GEOMETRY_HASH_COL in out.columns:
            prev_geom = str(out.loc[mask, GEOMETRY_HASH_COL].iloc[0] or "").strip()
        geom_ok = (
            mca_is_set(new_geom)
            and mca_is_set(prev_geom)
            and prev_geom.lower() == new_geom.lower()
        )
        if mca_is_set(prev_tok) and geom_ok:
            res_tok = str(prev_tok).strip()
    row = {
        **{k: key[k] for k in keys},
        **{c: shape.get(c, np.nan) for c in SHAPE_FINGERPRINT_COLS},
        GEOMETRY_HASH_COL: shape.get(GEOMETRY_HASH_COL, ""),
        "mca_segment": label,
        "label_status": status,
        RESOLVED_CONFLICT_TOKEN_COL: res_tok,
        "labeled_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_origin": origin,
    }
    if len(out) and int(mask.sum()) > 1:
        raise ValueError("Duplicate manual-label key; resolve duplicates before editing")
    if len(out) and mask.any():
        for col, val in row.items():
            out.loc[mask, col] = val
    else:
        out.loc[len(out)] = {col: row.get(col, np.nan) for col in out.columns}
    return out


def load_manual_labels(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=list(MANUAL_LABEL_COLUMNS))
    return normalize_id_columns(pd.read_csv(path, low_memory=False))


def save_manual_labels(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in MANUAL_LABEL_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    atomic_to_csv(df[cols + extra] if cols else df, path, index=False)


def analysis_revision_from_df(df: pd.DataFrame) -> str:
    if "analysis_revision" in df.columns:
        vals = df["analysis_revision"].dropna().astype(str).unique()
        if len(vals) == 1:
            return str(vals[0])
        if len(vals) > 1:
            return "|".join(sorted(vals))
    return ""
