"""Recording-group splits, train-only scaling, checkpoints, and evaluation."""
from __future__ import annotations

import csv
import json
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .metrics import classification_metrics
from .model import ELESON, objective


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_manifest(cache):
    return json.loads((Path(cache)/"manifest.json").read_text(encoding="utf-8"))


def make_splits(manifest, folds=5, seed=42, protocol="recording", holdout_mall=None, sources=None):
    records = manifest["records"]
    chosen = [i for i,r in enumerate(records) if not sources or r["source"] in sources]
    groups = defaultdict(list)
    for i in chosen:
        groups[records[i]["group"]].append(i)
    if len(groups) < folds:
        raise ValueError(f"need at least {folds} independent recordings, got {len(groups)}")
    rng = np.random.default_rng(seed)
    names = list(groups)
    rng.shuffle(names)
    # Greedy stratified group assignment: duplicated captures always stay together.
    hist = {g:np.bincount([y for i in groups[g] for y in records[i]["labels"]], minlength=3) for g in names}
    names.sort(key=lambda g: -int(hist[g].sum()))
    total = np.maximum(sum(hist.values()), 1)
    bins = np.zeros((folds, 3))
    assignment = {}
    for j, name in enumerate(names):
        scores = []
        for fold in range(folds):
            trial = bins.copy()
            trial[fold] += hist[name]
            scores.append(float(np.square(trial/total).sum()))
        fold = j if j < folds else int(np.argmin(scores))
        bins[fold] += hist[name]
        assignment[name] = fold
    result = []
    for fold in range(folds if protocol == "recording" else 1):
        if protocol == "mall":
            if not holdout_mall:
                raise ValueError("mall protocol requires --holdout-mall")
            test_groups = {g for g, ids in groups.items() if any(records[i]["mall"] == holdout_mall for i in ids)}
            if not test_groups:
                raise ValueError(f"no recordings in holdout mall {holdout_mall}")
        else:
            test_groups = {g for g in names if assignment[g] == fold}
        rest = [g for g in names if g not in test_groups]
        local = np.random.default_rng(seed+fold+1000)
        local.shuffle(rest)
        val_count = max(1, round(len(rest)*.1))
        val_groups = set(rest[:val_count])
        train_groups = set(rest[val_count:])
        if not train_groups:
            raise ValueError("not enough groups for train/validation/test")
        splits = {"train":sorted(i for g in train_groups for i in groups[g]),
                  "val":sorted(i for g in val_groups for i in groups[g]),
                  "test":sorted(i for g in test_groups for i in groups[g])}
        result.append(splits)
    return result


def fit_scaler(cache, manifest, record_ids, normalization="standard"):
    if normalization == "none":
        return {"mean":[0.]*7, "std":[1.]*7, "count":0, "normalization":"none"}
    count, sums, squares = 0, np.zeros(7), np.zeros(7)
    for i in record_ids:
        with np.load(Path(cache)/manifest["records"][i]["cache"]) as f:
            x = f["x"].astype(np.float64).reshape(-1, 7)
        count += len(x)
        sums += x.sum(0)
        squares += np.square(x).sum(0)
    if not count:
        raise ValueError("empty training split")
    mean = sums/count
    std = np.sqrt(np.maximum(squares/count-mean**2, 1e-8))
    return {"mean":mean.tolist(), "std":std.tolist(), "count":count, "normalization":"standard"}


class WindowDataset(Dataset):
    def __init__(self, cache, manifest, record_ids, scaler):
        self.cache, self.manifest = Path(cache), manifest
        self.rows = [(i,j) for i in record_ids for j in range(len(manifest["records"][i]["labels"]))]
        self.mean = np.asarray(scaler["mean"], dtype=np.float32)
        self.std = np.asarray(scaler["std"], dtype=np.float32)
        self.loaded = OrderedDict()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        i, j = self.rows[idx]
        r = self.manifest["records"][i]
        if i not in self.loaded:
            with np.load(self.cache/r["cache"]) as f:
                self.loaded[i] = f["x"]
            if len(self.loaded) > 16:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(i)
        x = (self.loaded[i][j]-self.mean)/self.std
        return torch.from_numpy(x), r["labels"][j], r["behavior"][j], idx


def predict(model, loader, device):
    model.eval()
    ys, predictions, confidence, uncertainty, indices = [], [], [], [], []
    with torch.no_grad():
        for x,y,_,idx in loader:
            out = model(x.to(device))
            conf, pred = out["confidence"].max(-1)
            ys.extend(y.tolist())
            predictions.extend(pred.cpu().tolist())
            confidence.extend(conf.cpu().tolist())
            uncertainty.extend(out["uncertainty"].cpu().tolist())
            indices.extend(idx.tolist())
    return {"y":np.array(ys), "pred":np.array(predictions), "confidence":np.array(confidence),
            "uncertainty":np.array(uncertainty), "index":np.array(indices)}


def score_predictions(p):
    return {str(t):classification_metrics(p["y"], p["pred"], p["confidence"], t) for t in (0., .5)}


def write_predictions(path, p, dataset):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["recording", "window", "start_ms", "label", "prediction", "confidence", "uncertainty", "source", "behavior"])
        for k, idx in enumerate(p["index"]):
            i,j = dataset.rows[int(idx)]
            r = dataset.manifest["records"][i]
            writer.writerow([r["path"], j, r["starts_ms"][j], int(p["y"][k]), int(p["pred"][k]),
                float(p["confidence"][k]), float(p["uncertainty"][k]), r["source"], r["behavior"][j]])


def train_fold(args, manifest, split, fold):
    seed_all(args.seed+fold)
    outdir = Path(args.output)/f"fold_{fold}"
    outdir.mkdir(parents=True, exist_ok=True)
    scaler = fit_scaler(args.cache, manifest, split["train"], args.normalization)
    datasets = {s:WindowDataset(args.cache, manifest, ids, scaler) for s,ids in split.items()}
    loaders = {s:DataLoader(ds, batch_size=args.batch_size, shuffle=s == "train",
                           num_workers=args.workers, generator=torch.Generator().manual_seed(args.seed+fold))
               for s,ds in datasets.items()}
    device = torch.device(args.device)
    samples = round(manifest["config"]["hz"]*manifest["config"]["seconds"])
    model_options = dict(feature_dim=128, hidden=args.hidden, frame_samples=args.frame_samples,
                         noise_std=args.noise_std, mode=args.model, features=args.features)
    if args.model_config:
        model_options.update(json.loads(Path(args.model_config).read_text(encoding="utf-8")))
    if "samples" in model_options:
        raise ValueError("samples is derived from the cache; do not specify it in model-config")
    model = ELESON(samples=samples, **model_options).to(device)
    loss_options = dict(w1=args.w1,w2=args.w2,w3=args.w3,w4=args.w4,
        causal=not args.no_causal, adversarial=not args.no_adversarial,
        uncertainty=not args.no_uncertainty, con_reduction="sample", adv_reduction="batch", variance_weight=1.)
    if args.loss_config:
        loss_options.update(json.loads(Path(args.loss_config).read_text(encoding="utf-8")))
    import inspect
    inspect.signature(objective).bind(None, None, None, None, **loss_options)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    training = vars(args).copy()
    training.update(scaler=scaler, fold=fold, split=split, preprocessing=manifest["config"],
                    model_options=model.config, loss_options=loss_options,
                    split_paths={s:[manifest["records"][i]["path"] for i in ids] for s,ids in split.items()},
                    parameters=sum(p.numel() for p in model.parameters()))
    (outdir/"config.json").write_text(json.dumps(training, indent=2), encoding="utf-8")
    best, stale, history = -1., 0, []
    for epoch in range(args.epochs):
        model.train()
        totals = defaultdict(float)
        seen = 0
        for step,(x,y,b,_) in enumerate(loaders["train"]):
            x,y,b = x.to(device),y.to(device),b.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x, auxiliary=True)
            loss, terms = objective(output, x,y,b, **loss_options)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss in fold {fold}, epoch {epoch+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            seen += len(y)
            for k,v in terms.items():
                totals[k] += float(v)*len(y)
            if args.max_steps and step+1 >= args.max_steps:
                break
        validation = predict(model, loaders["val"], device)
        metrics = classification_metrics(validation["y"], validation["pred"], validation["confidence"])
        row = dict(epoch=epoch+1, losses={k:v/seen for k,v in totals.items()}, validation=metrics)
        history.append(row)
        (outdir/"history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"fold={fold} epoch={epoch+1} loss={totals['optimization_loss']/seen:.4f} val_F1={metrics['conveyor_f1']:.4f}", flush=True)
        if metrics["conveyor_f1"] > best:
            best, stale = metrics["conveyor_f1"], 0
            torch.save(dict(model=model.state_dict(), model_config=model.config, scaler=scaler,
                            preprocessing=manifest["config"], split=split, epoch=epoch+1,
                            validation=metrics, optimizer=optimizer.state_dict(), loss_options=loss_options,
                            split_paths=training["split_paths"]), outdir/"best.pt")
        else:
            stale += 1
        if args.patience and stale >= args.patience:
            break
    checkpoint = torch.load(outdir/"best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    predictions = predict(model, loaders["test"], device)
    scores = score_predictions(predictions)
    (outdir/"metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    write_predictions(outdir/"predictions.csv", predictions, datasets["test"])
    return scores, predictions


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.folds < 3:
        raise ValueError("epochs and batch size must be positive; at least 3 folds are needed for train/val/test")
    torch.set_num_threads(args.threads)
    manifest = read_manifest(args.cache)
    splits = make_splits(manifest, args.folds, args.seed, args.protocol, args.holdout_mall, args.sources)
    folds = list(range(len(splits))) if args.fold == "all" else [int(args.fold)]
    if any(f < 0 or f >= len(splits) for f in folds):
        raise ValueError("requested fold is out of range")
    scores, pooled = {}, []
    Path(args.output).mkdir(parents=True, exist_ok=True)
    for fold in folds:
        result, prediction = train_fold(args, manifest, splits[fold], fold)
        scores[str(fold)] = result
        pooled.append(prediction)
    merged = {k:np.concatenate([p[k] for p in pooled]) for k in pooled[0]}
    aggregate = dict(folds=scores, pooled=score_predictions(merged),
        mean_f1={t:float(np.mean([s[t]["conveyor_f1"] for s in scores.values()])) for t in ("0.0", "0.5")},
        std_f1={t:float(np.std([s[t]["conveyor_f1"] for s in scores.values()])) for t in ("0.0", "0.5")})
    (Path(args.output)/"summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate["mean_f1"], indent=2))


def evaluate(args):
    torch.set_num_threads(args.threads)
    manifest = read_manifest(args.cache)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if manifest["config"] != checkpoint["preprocessing"]:
        raise ValueError("cache preprocessing differs from checkpoint")
    model = ELESON(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    if args.split == "all":
        ids = list(range(len(manifest["records"])))
    else:
        # Resolve by path so a regenerated/reordered manifest cannot silently
        # change the held-out recording set.
        split_paths = checkpoint.get("split_paths")
        if split_paths is None:  # compatibility with first-pass checkpoints
            cfg_path = Path(args.checkpoint).parent/"config.json"
            split_paths = json.loads(cfg_path.read_text(encoding="utf-8"))["split_paths"]
        paths = set(split_paths[args.split])
        ids = [i for i,r in enumerate(manifest["records"]) if r["path"] in paths]
        if len(ids) != len(paths):
            raise ValueError("checkpoint split contains recordings missing from cache")
    dataset = WindowDataset(args.cache, manifest, ids, checkpoint["scaler"])
    if not len(dataset):
        raise ValueError("empty evaluation set")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    prediction = predict(model, loader, device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scores = score_predictions(prediction)
    (output/"metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    write_predictions(output/"predictions.csv", prediction, dataset)
    print(json.dumps(scores, indent=2))
