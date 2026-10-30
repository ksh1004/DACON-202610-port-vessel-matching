"""Train byte-name transduction and compare a fixed residual with actual v12."""
import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import Adafactor, AutoModelForSeq2SeqLM, AutoTokenizer

from src.byte_name_model import MODEL_DIR, ByteNameScorer, encode, fragments, normalise, probability, sequence_logp, tensors
from src.common import ROOT, is_hangul
from src.pair_views import mix_views
from tools.blend_matching import lightweight_lab
from tools.prepare_byte_names import FOLDER, PROTOCOL, TAG, prepare

EPOCHS = (1, 3, 6, 12)
WEIGHT = .3


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_lock():
    p = json.loads(PROTOCOL.read_text())
    for path, expected in p["source_hashes"].items():
        assert sha(ROOT/path) == expected, path
    return p


def fingerprint(): return sha(PROTOCOL)


def dump(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    joblib.dump(value, temp, compress=3); temp.replace(path)


def seed():
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False


def model_loss(model, encoded, rows, device="cuda"):
    inputs, labels = tensors(encoded, device)
    values, valid = sequence_logp(model(**inputs, use_cache=False).logits, labels)
    losses = -values.sum(1) / valid.sum(1)
    weights = torch.tensor([r["weight"] for r in rows], dtype=torch.float32, device=device)
    return (losses * weights).sum() / weights.sum(), losses


@torch.inference_mode()
def validate(model, tokenizer, rows):
    model.eval(); total = count = 0.
    for offset in range(0, len(rows), 16):
        part = rows[offset:offset+16]
        encoded = [encode(tokenizer, (r["query"], r["target"])) for r in part]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, values = model_loss(model, encoded, part)
        weights = np.array([r["weight"] for r in part])
        total += float(np.dot(values.float().cpu().numpy(), weights)); count += float(weights.sum())
    return total/count


def train_phase(fold, phase, data, stop_epoch):
    check_lock(); seed()
    out = FOLDER / f"fold{fold}_{phase}"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "training.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        assert report["fingerprint"] == fingerprint()
        return report
    rows = data["fit"] if phase == "pilot" else data["full"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    encoded = [encode(tokenizer, (r["query"], r["target"])) for r in rows]
    checkpoints = []
    for slot in out.glob("checkpoint_slot*"):
        ready = slot / "ready.json"
        if ready.exists(): checkpoints.append((json.loads(ready.read_text())["epoch"], slot))
    state = None
    if checkpoints:
        completed, source = max(checkpoints)
        state = torch.load(source/"state.pt", map_location="cpu", weights_only=False)
        assert state["fingerprint"] == fingerprint() and state["data_sha256"] == data["data_sha256"]
    else:
        completed, source = 0, MODEL_DIR
    model = AutoModelForSeq2SeqLM.from_pretrained(source, local_files_only=True, dtype=torch.float32,
                                                attn_implementation="eager").to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = Adafactor(model.parameters(), lr=1e-3, scale_parameter=False, relative_step=False,
                         warmup_init=False, weight_decay=.01)
    history, best_epoch, best_nll, last_loss = [], None, float("inf"), None
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
        history, best_epoch, best_nll = state["history"], state["best_epoch"], state["best_nll"]
        last_loss = state.get("last_loss")
        del state
    per_epoch = math.ceil(len(rows)/16); horizon = per_epoch*12; warmup = math.ceil(horizon*.1)
    start = time.perf_counter(); loss = values = None
    for epoch in range(completed+1, stop_epoch+1):
        model.train(); losses = []
        order = np.random.default_rng(42+epoch).permutation(len(rows))
        for offset in range(0, len(order), 16):
            indices = order[offset:offset+16]
            step = (epoch-1)*per_epoch + offset//16
            factor = (step+1)/warmup if step < warmup else max((horizon-step)/(horizon-warmup), 0.)
            optimizer.param_groups[0]["lr"] = 1e-3 * factor
            part = [rows[int(i)] for i in indices]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, values = model_loss(model, [encoded[int(i)] for i in indices], part)
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite byte-name training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm): raise FloatingPointError("Nonfinite byte-name gradients")
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
        last_loss = float(np.mean(losses))
        if phase == "pilot" and epoch in EPOCHS:
            nll = validate(model, tokenizer, data["valid"])
            if not math.isfinite(nll): raise FloatingPointError("Nonfinite inner validation NLL")
            history.append({"epoch": epoch, "validation_token_nll": nll})
            if nll < best_nll:
                best_epoch, best_nll = epoch, nll
                destination = out/f"selected_epoch{epoch}"
                model.save_pretrained(destination); tokenizer.save_pretrained(destination)
            print(f"byte fold{fold} pilot epoch{epoch} train{last_loss:.4f} validation{nll:.4f} selected{best_epoch}", flush=True)
        else:
            print(f"byte fold{fold} {phase} epoch{epoch}/{stop_epoch} train{last_loss:.4f} seconds{time.perf_counter()-start:.1f}", flush=True)
        slot = out/f"checkpoint_slot{epoch%2}"
        slot.mkdir(exist_ok=True)
        (slot/"ready.json").unlink(missing_ok=True)
        model.save_pretrained(slot); tokenizer.save_pretrained(slot)
        state = {"epoch": epoch, "fingerprint": fingerprint(), "data_sha256": data["data_sha256"],
                 "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                 "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                 "history": history, "best_epoch": best_epoch, "best_nll": best_nll, "last_loss": last_loss}
        torch.save(state, slot/"state.pt.tmp"); (slot/"state.pt.tmp").replace(slot/"state.pt")
        (slot/"ready.json").write_text(json.dumps({"epoch": epoch}), encoding="utf-8")
        del state
    destination = out/"model"
    if phase == "full":
        model.eval().save_pretrained(destination); tokenizer.save_pretrained(destination)
    report = {"fingerprint": fingerprint(), "fold": fold, "phase": phase, "epochs": stop_epoch,
              "selected_epoch": best_epoch if phase == "pilot" else stop_epoch,
              "validation_history": history, "examples": len(rows), "steps": stop_epoch*per_epoch,
              "last_training_token_nll": last_loss, "resumed_after_epoch": completed,
              "seconds_this_segment": time.perf_counter()-start, "all_losses_and_gradients_finite": True,
              "data_sha256": data["data_sha256"], "deployment_dtype": "float32"}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    del model, optimizer, loss, values
    gc.collect(); torch.cuda.empty_cache()
    return report


def evaluation(fold):
    path = FOLDER/f"fold{fold}_evaluation.joblib"
    if path.exists(): return joblib.load(path)
    archived = joblib.load(ROOT/"experiments/encoder_bge_phonetic_predictions.joblib")
    lab = lightweight_lab(archived["identity"])
    frames = {"oof": lab.ctx.train, **{s: d.df for s, d in lab.sets.items()}}
    records = {s: dict(zip(df.sample_id, df.to_dict("records"))) for s, df in frames.items()}
    positions = {s: {sid: i for i, sid in enumerate(archived["identity"]["sets"][s]["sample_ids"])} for s in frames}
    old = joblib.load(ROOT/f"experiments/artifacts/encoder_bge_phonetic/fold{fold}_epoch1_scores.joblib")
    qwen = joblib.load(ROOT/f"experiments/artifacts/neural_evidence_v1/fold{fold}_epoch1_scores.joblib")
    from src.common import load_catalog
    catalog = load_catalog(); sets = {}
    for source, rows in old["sets"].items():
        sets[source] = []
        for i, g in enumerate(rows):
            sid = old["sample_ids"][source][i]; row = records[source][sid]
            keep = g["keep"]; candidates = g["candidates"][keep]
            bp = normalise(g["base_scores"][keep])
            if is_hangul(row["vessel_name"]) and not g["bound_pruned"]:
                anchor = normalise(mix_views(bp, [qwen["sets"][source][i]["neural_scores"], g["neural_scores"]], [1, 1]))
            else: anchor = bp
            assert int(candidates[np.argmax(anchor)]) == int(archived["ensemble"][source][positions[source][sid]])
            active = bool(is_hangul(row["vessel_name"]) and len(anchor)>1 and
                          np.sort(anchor)[-1]-np.sort(anchor)[-2] <= WEIGHT/(1-WEIGHT)+1e-6)
            sets[source].append({"sample_id": sid, "name": row["vessel_name"], "candidates": candidates,
                                 "names": catalog.vessel_name.iloc[candidates].tolist(), "anchor": anchor,
                                 "truth": int(g["truth"]), "active": active, "actual_hangul": bool(is_hangul(row["vessel_name"]))})
    result = {"sets": sets, "fingerprint": fingerprint()}; dump(result, path)
    return result


def score_cache(fold, variant, model_dir, data):
    check_lock()
    path = FOLDER/f"fold{fold}_{variant}_pairs.joblib"
    identity = {"config": sha(Path(model_dir)/"config.json"),
                "weights": {p.name: sha(p) for p in sorted(Path(model_dir).glob("*.safetensors")) + sorted(Path(model_dir).glob("pytorch_model*.bin"))}}
    cache = joblib.load(path) if path.exists() else {"fingerprint": fingerprint(), "model": identity, "scores": {}}
    assert cache["fingerprint"] == fingerprint()
    assert cache["model"] == identity
    pairs = sorted({(g["name"], s) for rows in data["sets"].values() for g in rows if g["active"]
                    for name in g["names"] for s, p in fragments(name)})
    missing = [p for p in pairs if p not in cache["scores"]]
    if not pairs: return {}
    model = ByteNameScorer(model_dir)
    # Verify the frozen per-pair batching recipe on this saved model before reuse.
    probe = missing[:17] if missing else pairs[:17]
    first = model.score_pairs(probe)
    np.testing.assert_array_equal(first, model.score_pairs(probe[::-1])[::-1])
    for i in (0, len(probe)//2, len(probe)-1):
        np.testing.assert_array_equal(model.score_pairs([probe[i]]), first[i:i+1])
    start = time.perf_counter()
    for offset in range(0, len(missing), 256):
        chunk = missing[offset:offset+256]
        cache["scores"].update(zip(chunk, model.score_pairs(chunk)))
        if offset%2048 == 0 or offset+256 >= len(missing):
            dump(cache, path)
            print(f"byte fold{fold} {variant} pairs{min(offset+256,len(missing))}/{len(missing)} seconds{time.perf_counter()-start:.1f}", flush=True)
    cache["invariance_probe_passed"] = True
    dump(cache, path)
    del model; gc.collect(); torch.cuda.empty_cache()
    return cache["scores"]


def results(fold, data, trained, control):
    result = {"fold": fold, "sets": {}, "controls": {}, "fingerprint": fingerprint()}
    for source, rows in data["sets"].items():
        for recipe in ("sets", "controls"): result[recipe][source] = []
        for g in rows:
            for recipe, cache in (("sets", trained), ("controls", control)):
                scores = None; prediction = None
                if g["actual_hangul"]:
                    values = g["anchor"]
                    if g["active"]:
                        scores = np.array([np.logaddexp.reduce([cache[(g["name"], s)] + math.log(p)
                                                               for s, p in fragments(name)]) for name in g["names"]])
                        values = (1-WEIGHT)*values + WEIGHT*probability(scores)
                    prediction = int(g["candidates"][np.argmax(values)])
                result[recipe][source].append({"sample_id": g["sample_id"], "truth": g["truth"],
                       "predictions": {"residual30": prediction}, "scores": scores,
                       "truth_in_top20": bool(g["truth"] in g["candidates"])})
    dump(result, FOLDER/f"fold{fold}_results.joblib")


def run(fold, train_only=False):
    check_lock(); data = prepare(fold)
    assert len(data["full"]) >= 500 and len(data["fit"]) >= 400 and len(data["valid"]) >= 100
    pilot = train_phase(fold, "pilot", data, 12)
    train_phase(fold, "full", data, pilot["selected_epoch"])
    if train_only: return
    evaluation_data = evaluation(fold)
    assert evaluation_data["fingerprint"] == fingerprint()
    trained = score_cache(fold, "trained", FOLDER/f"fold{fold}_full/model", evaluation_data)
    control = score_cache(fold, "untrained", MODEL_DIR, evaluation_data)
    results(fold, evaluation_data, trained, control)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--folds", type=int, nargs="+", default=[1, 4])
    p.add_argument("--train-only", action="store_true"); args = p.parse_args()
    for fold in args.folds: run(fold, args.train_only)
