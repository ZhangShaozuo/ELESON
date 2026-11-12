"""Raw CSV -> aligned, labeled 100 Hz windows. All time arithmetic uses milliseconds."""
from __future__ import annotations

import csv
import hashlib
import json
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

CLASSES = ["neither", "elevator", "escalator"]
SOURCES = ["Single_Label_Data", "Segmented_Data", "Trajectory_Data"]


def sensor(path: Path):
    with path.open(encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    try:
        a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except ValueError:
        # A released long capture contains a truncated CSV row. Drop malformed
        # rows, retain their actual timestamps, then let the gap mask decide.
        dropped = [0]
        def good_lines():
            with path.open(encoding="utf-8-sig") as f:
                next(f)
                for line in f:
                    if line.count(",") == len(header)-1:
                        yield line
                    else:
                        dropped[0] += 1
        a = np.loadtxt(good_lines(), delimiter=",", ndmin=2)
        if dropped[0]:
            warnings.warn(f"dropped {dropped[0]} malformed sensor rows: {path}")
    if a.shape[1] != len(header) or len(a) < 2:
        raise ValueError(f"invalid sensor shape: {path}")
    # imuTimestamp is the actual sample clock; elapsedRealtime in old files is
    # a callback clock and can repeat for a whole delivered batch.
    clock = "imuTimestamp" if "imuTimestamp" in header else "elapsedRealtime"
    t = a[:, header.index(clock)]
    xyz = a[:, -3:]
    valid = np.isfinite(t) & np.isfinite(xyz).all(axis=1)
    t, xyz = t[valid], xyz[valid]
    order = np.argsort(t, kind="stable")
    t, xyz = t[order], xyz[order]
    unique = np.r_[True, np.diff(t) > 0]
    offset = None
    if "currentTimeMillis" in header:
        offset = float(np.nanmedian(a[:, header.index("currentTimeMillis")] -
                                    a[:, header.index("elapsedRealtime")]))
    t, xyz = t[unique], xyz[unique]
    if len(t) < 2:
        raise ValueError(f"fewer than two unique timestamps: {path}")
    return t, xyz, offset


def trajectory_labels(root: Path):
    """Merge the two block-form annotation files; conflicting ticks stay unknown."""
    annotations, conflicts = {}, set()
    for file in sorted((root / "Trajectory_Data").glob("Label_file_*.csv")):
        with file.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        for i, row in enumerate(rows):
            if not row or row[0].strip() != "traj_id":
                continue
            ident = row[1].strip()
            if i + 2 >= len(rows) or rows[i+1][0] != "relative_time" or rows[i+2][0] != "label":
                raise ValueError(f"malformed label block {file}:{i+1}")
            dest = annotations.setdefault(ident, {})
            for ts, label in zip(rows[i+1][1:], rows[i+2][1:]):
                if not ts.strip() or not label.strip():
                    continue
                tick, raw = float(ts), int(float(label))
                value = {0: 0, 1: 0, 3: 1, 4: 2}.get(raw, -1)
                key = (ident, tick)
                if key in conflicts or (tick in dest and dest[tick] != value):
                    conflicts.add(key)
                    dest[tick] = -1
                else:
                    dest[tick] = value
    result = {}
    for ident, values in annotations.items():
        t = np.array(sorted(values))
        y = np.array([values[v] for v in t])
        # Every timestamp labels the following 1.28 s bin. Infer this duration
        # from the file instead of extending the final label indefinitely.
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1280.0
        result[ident] = (t, y, dt)
    return result, len(conflicts)


def recording_labels(root, path, grid, acc_start, offset, annotations, boundary, trajectory_offset_ms=0.):
    rel = path.relative_to(root)
    source = rel.parts[0]
    if source == "Single_Label_Data":
        return np.full(len(grid), CLASSES.index(rel.parts[1].lower()), dtype=np.int64)
    if source == "Trajectory_Data":
        ident = path.parent.name
        if ident not in annotations:
            raise ValueError("trajectory has no annotation block")
        ts, values, dt = annotations[ident]
        relative = grid - acc_start - trajectory_offset_ms
        idx = np.searchsorted(ts, relative, side="right") - 1
        safe = np.clip(idx, 0, len(ts)-1)
        return np.where((idx >= 0) & (relative < ts[safe]+dt), values[safe], -1)
    decoded = []
    for part in rel.parts:
        try:
            part = part.encode("cp850").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        decoded.append(part)
    text = "/".join(decoded).lower()
    target = 2 if "escalator" in text or "扶梯" in text else 1 if "elevator" in text or "电梯" in text else 0
    if target == 0:
        if "walking" not in text:
            raise ValueError("cannot infer segmented state from path")
        return np.zeros(len(grid), dtype=np.int64)
    label_file = path.parent / "speak.csv"
    if not label_file.exists() or offset is None:
        raise ValueError("conveyor recording missing speak.csv or epoch clock bridge")
    # Only the conveyor interval is guaranteed by the released readme.
    # Outside it remains unknown unless the user explicitly opts in.
    y = np.full(len(grid), 0 if boundary == "neither" else -1, dtype=np.int64)
    events = {"cross-layer label": [], "still label": []}
    with label_file.open(encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row or not row[-1].strip():
                continue
            kind = row[0].strip()
            if kind not in events:
                raise ValueError("unrecognized/damaged speak event label")
            events[kind].append(float(row[-1])-offset)
    assigned = np.zeros(len(grid), dtype=bool)
    any_overlap = False
    for kind, times in events.items():
        # There may be many rides plus explicitly annotated standing intervals.
        if len(times) % 2 or any(b <= a for a,b in zip(times, times[1:])):
            raise ValueError("speak event timestamps are unpaired or unordered")
        for start,end in zip(times[::2], times[1::2]):
            mask = (grid >= start) & (grid < end)
            if (assigned & mask).any():
                raise ValueError("overlapping speak intervals")
            any_overlap |= bool(mask.any())
            y[mask] = target if kind == "cross-layer label" else 0
            assigned |= mask
    if not any_overlap:
        raise ValueError("speak intervals do not overlap sensor timestamps")
    return y


def align_recording(root, path, annotations, *, hz=100, max_gap_ms=100,
                    boundary="unknown", frame="device", diff_start="drop",
                    acc_scale=9.80665, gravity_tau=.5, trajectory_offset_ms=0.):
    streams = []
    for name in ("acc", "gyro", "mag"):
        candidates = sorted(path.parent.glob(name + "_*.csv"))
        candidates = [p for p in candidates if not p.name.startswith("._")]
        if name == "acc":
            chosen = path
        elif len(candidates) == 1:
            chosen = candidates[0]
        elif len(candidates) > 1:
            def timestamp(p):
                return datetime.strptime(p.stem.split("_", 1)[1], "%Y_%m_%d_%H_%M_%S")
            skew = [abs((timestamp(p)-timestamp(path)).total_seconds()) for p in candidates]
            best = int(np.argmin(skew))
            if skew[best] > 60 or skew.count(skew[best]) > 1:
                raise ValueError(f"ambiguous {name} file pairing")
            chosen = candidates[best]
        else:
            raise ValueError(f"missing {name} sensor")
        streams.append(sensor(chosen))
    lo, hi = max(s[0][0] for s in streams), min(s[0][-1] for s in streams)
    if hi <= lo:
        raise ValueError("sensor clocks have no common overlap")
    grid = np.arange(lo, hi + 1e-5, 1000 / hz, dtype=np.float64)
    valid = np.ones(len(grid), dtype=bool)
    xyz = []
    for t, v, _ in streams:
        xyz.append(np.stack([np.interp(grid, t, v[:, j]) for j in range(3)], axis=1))
        right = np.searchsorted(t, grid, side="left")
        left = np.maximum(right-1, 0)
        right = np.minimum(right, len(t)-1)
        exact = np.isclose(grid, t[right], rtol=0, atol=1e-4)
        valid &= exact | ((t[right]-t[left]) <= max_gap_ms)
    acc, gyro, mag = xyz
    raw_gyro_norm = np.linalg.norm(gyro, axis=1)
    if frame == "gravity":
        # Approximate gravity alignment, not RoNIN's quaternion preprocessing.
        gravity = acc.copy()
        alpha = np.exp(-1 / hz / gravity_tau)
        for i in range(1, len(gravity)):
            gravity[i] = (alpha * gravity[i-1] + (1-alpha) * acc[i]
                          if valid[i] and valid[i-1] else acc[i])
        z = gravity / np.maximum(np.linalg.norm(gravity, axis=1, keepdims=True), 1e-6)
        ref = np.tile([1., 0., 0.], (len(z), 1))
        ref[np.abs(z[:, 0]) > .9] = [0., 1., 0.]
        x = ref - (ref*z).sum(1, keepdims=True)*z
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
        basis = np.stack([x, np.cross(z, x), z], axis=1)
        acc = np.einsum("nij,nj->ni", basis, acc)
        gyro = np.einsum("nij,nj->ni", basis, gyro)
    intensity = np.linalg.norm(mag, axis=1)
    diff = np.diff(intensity, prepend=intensity[0])
    # Differential is computed on a continuous recording, before windowing.
    valid[1:] &= valid[:-1].copy()
    features = np.column_stack([acc / acc_scale, gyro, diff]).astype(np.float32)
    labels = recording_labels(root, path, grid, streams[0][0][0], streams[0][2], annotations, boundary, trajectory_offset_ms)
    labels[~valid] = -1
    if diff_start == "drop":
        # Eq.21 requires two observations. Remove the fabricated zero and start
        # windowing one sample later, rather than dropping a whole 2 s window.
        features, labels, raw_gyro_norm, grid = features[1:], labels[1:], raw_gyro_norm[1:], grid[1:]
    # Same raw capture copied to different dataset folders must share a fold.
    t, a, _ = streams[0]
    fingerprint = hashlib.sha256(np.round(np.r_[t[0], t[-1], a[:8].ravel(), a[-8:].ravel()], 4).tobytes()).hexdigest()[:20]
    return features, labels, raw_gyro_norm, grid, fingerprint


def prepare(root: Path, output: Path, *, hz=100, seconds=2., stride=2.,
            purity=1., max_gap_ms=100., boundary="unknown", frame="device", limit=None,
            diff_start="drop", acc_scale=9.80665, gravity_tau=.5,
            trajectory_offset_ms=0., behavior_threshold=1.5, behavior_reducer="mean"):
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "recordings").mkdir(exist_ok=True)
    annotations, conflict_count = trajectory_labels(root)
    paths = sorted(p for s in SOURCES for p in (root / s).rglob("acc_*.csv") if not p.name.startswith("._"))
    if limit:
        # Round robin sources AND labels, so a smoke run covers every format.
        buckets = {}
        for p in paths:
            rel = p.relative_to(root)
            key = (rel.parts[0], rel.parts[1] if rel.parts[0] != "Trajectory_Data" else "")
            buckets.setdefault(key, []).append(p)
        paths = []
        while buckets and len(paths) < limit:
            for key in list(buckets):
                paths.append(buckets[key].pop(0))
                if not buckets[key]:
                    del buckets[key]
                if len(paths) >= limit:
                    break
    n, step = round(hz*seconds), round(hz*stride)
    if (hz <= 0 or n < 2 or step < 1 or max_gap_ms <= 0 or not 0 < purity <= 1
            or acc_scale <= 0 or gravity_tau <= 0 or behavior_threshold < 0
            or diff_start not in ("drop", "zero") or behavior_reducer not in ("mean", "rms", "max")):
        raise ValueError("invalid window parameters")
    report = {"discovered_recordings": len(paths), "annotation_conflict_ticks": conflict_count,
              "errors": [], "warnings": [], "dropped_windows": 0, "by_source": {}, "class_counts": {}}
    records, counts, sources = [], Counter(), Counter()
    for i, path in enumerate(paths):
        rel = path.relative_to(root)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                x, y, angular, grid, group = align_recording(root, path, annotations, hz=hz,
                    max_gap_ms=max_gap_ms, boundary=boundary, frame=frame, diff_start=diff_start,
                    acc_scale=acc_scale, gravity_tau=gravity_tau, trajectory_offset_ms=trajectory_offset_ms)
            report["warnings"].extend({"path":rel.as_posix(), "reason":str(w.message)} for w in caught)
            windows, labels, behaviors, starts = [], [], [], []
            for start in range(0, len(x)-n+1, step):
                ys = y[start:start+n]
                if (ys < 0).any():
                    report["dropped_windows"] += 1
                    continue
                label_counts = np.bincount(ys, minlength=3)
                target = int(label_counts.argmax())
                if label_counts[target] / n < purity:
                    report["dropped_windows"] += 1
                    continue
                windows.append(x[start:start+n])
                labels.append(target)
                # The paper specifies 1.5 rad/s, but not the temporal reducer.
                v = angular[start:start+n]
                level = v.mean() if behavior_reducer == "mean" else np.sqrt(np.square(v).mean()) if behavior_reducer == "rms" else v.max()
                behaviors.append(int(level > behavior_threshold))
                starts.append(float(grid[start]))
            if not windows:
                raise ValueError("no fully labeled valid windows")
            ident = hashlib.sha256(str(rel).encode()).hexdigest()[:20]
            cache = f"recordings/{ident}.npz"
            np.savez_compressed(output / cache, x=np.stack(windows))
            records.append(dict(path=rel.as_posix(), cache=cache, source=rel.parts[0],
                mall=rel.parts[1] if rel.parts[0] == "Segmented_Data" else None,
                group=group, labels=labels, behavior=behaviors, starts_ms=starts))
            counts.update(CLASSES[v] for v in labels)
            sources[rel.parts[0]] += len(labels)
        except (ValueError, OSError, IndexError) as exc:
            report["errors"].append({"path": rel.as_posix(), "reason": str(exc)})
        if (i+1) % 100 == 0:
            print(f"prepared {i+1}/{len(paths)} recordings", flush=True)
    report.update(usable_recordings=len(records), windows=sum(counts.values()),
                  class_counts=dict(counts), by_source=dict(sources))
    config = dict(hz=hz, seconds=seconds, stride=stride, purity=purity,
                  max_gap_ms=max_gap_ms, boundary=boundary, frame=frame, channels=7,
                  pipeline_version=3, diff_start=diff_start, acc_scale=acc_scale,
                  gravity_tau=gravity_tau, trajectory_offset_ms=trajectory_offset_ms,
                  behavior_threshold=behavior_threshold, behavior_reducer=behavior_reducer)
    (output / "manifest.json").write_text(json.dumps(dict(version=1, classes=CLASSES,
        config=config, records=records), indent=2), encoding="utf-8")
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("errors", "warnings")}, indent=2))
    print(f"Skipped recordings: {len(report['errors'])}; see {output / 'audit.json'}")
    if not records:
        raise RuntimeError("No usable recordings; inspect audit.json")
    return report
