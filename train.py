"""Train a small Transformer on prepared real text; save logs, checkpoints and loss curves."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import ModelConfig, TransformerLM


PROJECT = Path(__file__).resolve().parent


def sample_batch(tokens, batch_size, context_length, generator, device):
    if len(tokens) <= context_length or batch_size < 1:
        raise ValueError("Need a positive batch and more tokens than the context length.")
    starts = torch.randint(len(tokens) - context_length, (batch_size,), generator=generator)
    offsets = starts[:, None] + torch.arange(context_length)[None, :]
    return tokens[offsets].to(device), tokens[offsets + 1].to(device)


def load_data(data_dir, context_length):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    tokenizer_bytes = (data_dir / "tokenizer.json").read_bytes()
    tokenizer = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
    if tokenizer.get_vocab_size() != manifest["vocab_size"] or manifest["dtype"] != "<i4":
        raise ValueError("Data manifest does not match tokenizer or int32 format.")
    arrays = []
    for split in ["train", "val"]:
        raw = (data_dir / f"{split}.bin").read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest[f"{split}_bin_sha256"]:
            raise ValueError(f"{split}.bin hash mismatch; rerun data preparation or use the original data.")
        values = np.frombuffer(raw, dtype="<i4")
        if len(values) != manifest[f"{split}_tokens"] or len(values) <= context_length:
            raise ValueError(f"{split} token count is incorrect or too small for this context.")
        if values.min() < 0 or values.max() >= manifest["vocab_size"]:
            raise ValueError(f"{split} contains an out-of-vocabulary token ID.")
        original_text = (data_dir / f"{split}.txt").read_bytes().decode("utf-8")
        if tokenizer.decode(values.tolist(), skip_special_tokens=False) != original_text:
            raise ValueError(f"{split} text and IDs do not match this tokenizer; prepare data together again.")
        arrays.append(torch.from_numpy(values.astype(np.int64)))
    provenance = {key: manifest[key] for key in [
        "source_file", "source_sha256", "split_note", "seed", "train_bin_sha256", "val_bin_sha256"
    ]}
    provenance["tokenizer_sha256"] = hashlib.sha256(tokenizer_bytes).hexdigest()
    return arrays[0], arrays[1], manifest["vocab_size"], provenance


@torch.no_grad()
def evaluate(model, tokens, batch_size, context_length, batches, seed, device):
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(batches):
        x, y = sample_batch(tokens, batch_size, context_length, generator, device)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        total += loss.item()
    model.train(was_training)
    loss = total / batches
    if not math.isfinite(loss):
        raise RuntimeError("Validation loss is non-finite.")
    return {"val_loss": loss, "val_ppl": math.exp(loss)}


def save_checkpoint(path, model, optimizer, step, train_generator, provenance):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": 1, "config": asdict(model.config),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": step, "train_rng_state": train_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(), "provenance": provenance,
        "torch_version": str(torch.__version__),
    }
    torch.save(payload, temporary)
    temporary.replace(path)


def plot_losses(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4))
    for key, label in [("train_loss", "Training (recent steps)"), ("val_loss", "Validation (fixed sample)")]:
        chosen = [row for row in rows if row.get(key) is not None]
        axis.plot([row["step"] for row in chosen], [row[key] for row in chosen], label=label)
    axis.set(xlabel="Training step", ylabel="Cross-entropy loss", title="TinyStories teaching run")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    parser.add_argument("--out", type=Path, default=PROJECT / "checkpoints/quickstart")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def run_training(args):
    if min(args.steps, args.batch_size, args.eval_every, args.eval_batches, args.log_every, args.threads) < 1:
        raise ValueError("Step, batch, interval and thread counts must be positive.")
    if not math.isfinite(args.lr) or args.lr <= 0 or args.warmup_steps < 0:
        raise ValueError("Learning rate must be positive/finite and warmup nonnegative.")
    config = ModelConfig(1, args.d_model, args.heads, args.layers, args.context_length)
    train_tokens, val_tokens, vocab_size, provenance = load_data(args.data_dir, args.context_length)
    config = ModelConfig(vocab_size, config.d_model, config.num_heads, config.num_layers, config.context_length)
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {output}. Choose a new --out directory.")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable in this Python environment.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model = TransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    generator = torch.Generator().manual_seed(args.seed)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | Parameters: {parameter_count:,} | CPU threads: {args.threads}", flush=True)
    print(f"Steps: {args.steps} | Batch: {args.batch_size} | Context: {args.context_length}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    baseline = evaluate(model, val_tokens, args.batch_size, args.context_length, args.eval_batches, args.seed + 100, device)
    rows = [{"step": 0, "train_loss": None, **baseline, "elapsed_seconds": time.perf_counter() - started}]
    print(f"Initial validation loss: {baseline['val_loss']:.4f} | PPL: {baseline['val_ppl']:.2f}", flush=True)
    best_loss, best_step = float("inf"), None
    training_seconds = 0.0
    recent_losses = []
    warmup = min(args.warmup_steps, max(0, args.steps - 1))
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        log.write(json.dumps(rows[0]) + "\n")
        log.flush()
        for step in range(1, args.steps + 1):
            tick = time.perf_counter()
            if warmup and step <= warmup:
                lr = args.lr * step / warmup
            else:
                progress = (step - warmup) / max(1, args.steps - warmup)
                lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = sample_batch(train_tokens, args.batch_size, args.context_length, generator, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss is non-finite; use a lower learning rate.")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Gradient norm is non-finite.")
            optimizer.step()
            recent_losses.append(loss.item())
            if device == "cuda":
                torch.cuda.synchronize()
            training_seconds += time.perf_counter() - tick
            should_eval = step % args.eval_every == 0 or step == args.steps
            if should_eval or step % args.log_every == 0:
                row = {"step": step, "train_loss": sum(recent_losses) / len(recent_losses), "lr": lr}
                recent_losses.clear()
                if should_eval:
                    row.update(evaluate(model, val_tokens, args.batch_size, args.context_length, args.eval_batches, args.seed + 100, device))
                    save_checkpoint(output / "last.pt", model, optimizer, step, generator, provenance)
                    if row["val_loss"] < best_loss:
                        best_loss, best_step = row["val_loss"], step
                        save_checkpoint(output / "best.pt", model, optimizer, step, generator, provenance)
                row["elapsed_seconds"] = time.perf_counter() - started
                row["train_tokens_per_second"] = step * args.batch_size * args.context_length / training_seconds
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                text = f"Step {step}/{args.steps} | Train loss {row['train_loss']:.4f}"
                if should_eval:
                    text += f" | Val loss {row['val_loss']:.4f} | PPL {row['val_ppl']:.2f}"
                print(text + f" | {row['train_tokens_per_second']:.0f} tokens/s", flush=True)
    summary = {
        "steps": args.steps, "batch_size": args.batch_size, "seed": args.seed,
        "config": asdict(config), "parameter_count": parameter_count,
        "device": device, "threads": args.threads, "torch_version": str(torch.__version__),
        "baseline_val_loss": baseline["val_loss"], "best_val_loss": best_loss,
        "best_val_ppl": math.exp(best_loss), "best_step": best_step,
        "final_val_loss": rows[-1]["val_loss"], "final_train_loss": rows[-1]["train_loss"],
        "trained_tokens": args.steps * args.batch_size * args.context_length,
        "wall_seconds": time.perf_counter() - started, "train_seconds": training_seconds,
        "train_tokens_per_second": args.steps * args.batch_size * args.context_length / training_seconds,
        "validation_note": "Loss/PPL estimated on fixed randomly sampled held-out windows, not every validation token.",
        "provenance": provenance,
    }
    plot_losses(rows, output / "loss.png")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"TRAINING COMPLETE | Best step: {best_step} | Val PPL: {summary['best_val_ppl']:.2f}", flush=True)
    print(f"Artifacts: {output.resolve()}", flush=True)
    return summary


if __name__ == "__main__":
    run_training(build_parser().parse_args())
