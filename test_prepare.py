"""Behavior checks for the teaching data pipeline; run with unittest discover."""

import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from tokenizers import Tokenizer


class PrepareTests(unittest.TestCase):
    def api(self):
        # Missing implementation must fail explicitly before import.
        self.assertIsNotNone(importlib.util.find_spec("prepare"), "prepare.py is not implemented")
        return importlib.import_module("prepare").prepare

    def fixture(self, root):
        source = root / "source.txt"
        source.write_text(
            "One small dog.<|endoftext|>A child plays.<|endoftext|>"
            "One small dog.<|endoftext|>The sun is warm.<|endoftext|>"
            "A bird can fly.<|endoftext|><|endoftext|>", encoding="utf-8"
        )
        return source

    def test_duplicate_stories_cannot_leak_between_splits(self):
        prepare = self.api()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            info = prepare(source, root / "out", 3, 1, 384, 42)
            train = (root / "out/train.txt").read_text(encoding="utf-8").split("<|endoftext|>")
            val = (root / "out/val.txt").read_text(encoding="utf-8").split("<|endoftext|>")
            train, val = {s.strip() for s in train if s.strip()}, {s.strip() for s in val if s.strip()}
            self.assertEqual(len(train), 3)
            self.assertEqual(len(val), 1)
            self.assertFalse(train & val)
            self.assertEqual(train | val, {"One small dog.", "A child plays.", "The sun is warm.", "A bird can fly."})
            self.assertEqual(info["duplicates_removed"], 1)
            other = prepare(source, root / "other", 3, 1, 384, 42)
            self.assertEqual(info["train_document_sha256"], other["train_document_sha256"])
            self.assertEqual(info["val_document_sha256"], other["val_document_sha256"])

    def test_persisted_tokenizer_preserves_unseen_bytes_and_binary_boundaries(self):
        prepare = self.api()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = prepare(self.fixture(root), root / "out", 3, 1, 384, 42)
            tokenizer = Tokenizer.from_file(str(root / "out/tokenizer.json"))
            sample = "  汉字🚀\r\n\tＣafé"
            ids = tokenizer.encode(sample).ids
            self.assertEqual(tokenizer.decode(ids, skip_special_tokens=False), sample)
            self.assertNotIn(tokenizer.token_to_id("<|endoftext|>"), ids)
            for split, count in [("train", 3), ("val", 1)]:
                tokens = np.fromfile(root / f"out/{split}.bin", dtype="<i4")
                self.assertEqual(len(tokens), info[f"{split}_tokens"])
                self.assertEqual(int((tokens == tokenizer.token_to_id("<|endoftext|>")).sum()), count)
                self.assertGreaterEqual(int(tokens.min()), 0)
                self.assertLess(int(tokens.max()), tokenizer.get_vocab_size())
                self.assertEqual(
                    tokenizer.decode(tokens.tolist(), skip_special_tokens=False),
                    (root / f"out/{split}.txt").read_bytes().decode("utf-8"),
                )
            saved = json.loads((root / "out/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["dtype"], "<i4")
            self.assertEqual(saved["source_sha256"], info["source_sha256"])
            self.assertIn("not the official", saved["split_note"])

    def test_rejects_too_few_unique_documents_before_writing(self):
        prepare = self.api()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            with self.assertRaises(ValueError):
                prepare(source, root / "out", 4, 1, 384, 42)
            self.assertFalse((root / "out").exists())

    def test_rejects_vocabulary_too_small_for_byte_coverage(self):
        prepare = self.api()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                prepare(self.fixture(root), root / "out", 3, 1, 128, 42)


if __name__ == "__main__":
    unittest.main()
