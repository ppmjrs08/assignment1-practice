"""Prepare a small, reproducible TinyStories teaching dataset and byte-level BPE."""

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


PROJECT = Path(__file__).resolve().parent
EOS = "<|endoftext|>"


def prepare(source: Path, output: Path, train_docs: int = 1800,
            val_docs: int = 200, vocab_size: int = 2048, seed: int = 42) -> dict:
    """Split whole unique documents, train on train only, and serialize int32 IDs."""
    if train_docs < 1 or val_docs < 1:
        raise ValueError("Both train_docs and val_docs must be positive.")
    if vocab_size < 257:
        raise ValueError("vocab_size must hold 256 byte symbols and one special token (>=257).")

    raw = Path(source).read_bytes()
    documents = [part.strip() for part in raw.decode("utf-8").split(EOS) if part.strip()]
    unique = list(dict.fromkeys(documents))
    if len(unique) < train_docs + val_docs:
        raise ValueError(f"Need {train_docs + val_docs} unique documents; found {len(unique)}.")
    random.Random(seed).shuffle(unique)
    training = unique[:train_docs]
    validation = unique[train_docs:train_docs + val_docs]

    # Full byte alphabet covers characters never observed in the training stories.
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
        show_progress=False,
    )
    # No normalizer: preserve original spaces, width, accents and line breaks.
    # Validation documents are deliberately excluded from this call.
    tokenizer.train_from_iterator(training, trainer=trainer, length=len(training))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output / "tokenizer.json"))
    loaded = Tokenizer.from_file(str(output / "tokenizer.json"))

    info = {
        "schema_version": 1,
        "source_file": Path(source).name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "source_documents": len(documents),
        "unique_documents": len(unique),
        "duplicates_removed": len(documents) - len(unique),
        "split_note": "Teaching re-split of a local source pool; not the official train/validation benchmark split. Source file name is preserved above.",
        "seed": seed,
        "requested_vocab_size": vocab_size,
        "vocab_size": loaded.get_vocab_size(),
        "tokenizers_version": tokenizers.__version__,
        "dtype": "<i4",
        "eos_token": EOS,
        "eos_id": loaded.token_to_id(EOS),
        "normalization": "none",
        "add_prefix_space": False,
        "full_byte_alphabet": True,
    }

    for split, stories in [("train", training), ("val", validation)]:
        (output / f"{split}.txt").write_text(
            EOS.join(stories) + EOS, encoding="utf-8", newline="\n"
        )
        ids = []
        for story in stories:
            ids.extend(loaded.encode(story).ids)
            ids.append(info["eos_id"])
        values = np.asarray(ids, dtype="<i4")
        values.tofile(output / f"{split}.bin")
        info[f"{split}_documents"] = len(stories)
        info[f"{split}_tokens"] = len(values)
        info[f"{split}_bin_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
        info[f"{split}_document_sha256"] = [
            hashlib.sha256(story.encode("utf-8")).hexdigest() for story in stories
        ]

    samples = ["Hello, world!", "  我在学习 BPE 🚀\n", "Ｃafé\tPython"]
    checks = []
    for sample in samples:
        ids = loaded.encode(sample).ids
        restored = loaded.decode(ids, skip_special_tokens=False)
        if restored != sample or tokenizer.encode(sample).ids != ids:
            raise RuntimeError("Tokenizer failed a round-trip / reload check.")
        checks.append({"text": sample, "ids": ids, "roundtrip_ok": True})
    info["roundtrip_checks"] = checks
    (output / "manifest.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT.parent / "diy-llm/coursework/assignment1-basics/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--output", type=Path, default=PROJECT / "data")
    parser.add_argument("--train-docs", type=int, default=1800)
    parser.add_argument("--val-docs", type=int, default=200)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    info = prepare(args.source, args.output, args.train_docs, args.val_docs, args.vocab_size, args.seed)
    print("DATA AND BPE READY")
    print(f"Train stories: {info['train_documents']} | Validation stories: {info['val_documents']}")
    print(f"Vocabulary: {info['vocab_size']}")
    print(f"Train tokens: {info['train_tokens']} | Validation tokens: {info['val_tokens']}")
    print(f"Round-trip and reload checks: {len(info['roundtrip_checks'])}/3 PASS")
    print(f"Output: {args.output.resolve()}")
    print("Note: teaching re-split of local text, not the official benchmark split.")


if __name__ == "__main__":
    main()
