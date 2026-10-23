"""Load an actual checkpoint and generate text with its matching BPE tokenizer."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import ModelConfig, TransformerLM


PROJECT = Path(__file__).resolve().parent


def load_model(checkpoint, data_dir, device):
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema.")
    tokenizer_path = Path(data_dir) / "tokenizer.json"
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != payload["provenance"]["tokenizer_sha256"]:
        raise ValueError("tokenizer hash differs from the checkpoint; use the original training tokenizer.")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    config = ModelConfig(**payload["config"])
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match checkpoint model dimensions.")
    model = TransformerLM(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, tokenizer, payload


@torch.no_grad()
def generate_ids(model, prompt_ids, max_new_tokens=120, temperature=0.8,
                 top_k=40, eos_id=None, seed=42):
    if not prompt_ids or max_new_tokens < 1:
        raise ValueError("Provide a nonempty prompt and positive max_new_tokens.")
    if not math.isfinite(temperature) or temperature <= 0 or top_k < 1:
        raise ValueError("temperature must be positive/finite and top_k positive.")
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        context = ids[:, -model.config.context_length:]
        logits = model(context)[:, -1, :] / temperature
        top = min(top_k, logits.shape[-1])
        cutoff = torch.topk(logits, top).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, -torch.inf)
        probabilities = torch.softmax(logits, dim=-1)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Generation probabilities are non-finite.")
        next_token = torch.multinomial(probabilities, 1, generator=generator)
        ids = torch.cat((ids, next_token), dim=1)
        if eos_id is not None and next_token.item() == eos_id:
            break
    return ids[0].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "checkpoints/quickstart/best.pt")
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model, tokenizer, saved = load_model(args.checkpoint, args.data_dir, device)
    ids = generate_ids(model, tokenizer.encode(args.prompt).ids, args.max_new_tokens,
                       args.temperature, args.top_k, tokenizer.token_to_id("<|endoftext|>"), args.seed)
    text = tokenizer.decode(ids, skip_special_tokens=True)
    output = args.output or args.checkpoint.parent / "generated.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((text + "\n").encode("utf-8"))
    details = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_step": saved["step"],
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "prompt": args.prompt, "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature, "top_k": args.top_k, "seed": args.seed,
        "total_tokens": len(ids), "device": device,
    }
    output.with_suffix(".json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Loaded trained checkpoint: step {saved['step']} | Device: {device}")
    print("Generated text:")
    print(text)
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
