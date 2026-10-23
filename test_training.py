"""Checks for causal modeling, next-token targets and persisted training artifacts."""

import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F

from prepare import prepare


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def modules(self):
        for name in ["model", "train", "generate"]:
            self.assertIsNotNone(importlib.util.find_spec(name), f"{name}.py is not implemented")
        return tuple(importlib.import_module(name) for name in ["model", "train", "generate"])

    def data(self, root):
        source = root / "stories.txt"
        source.write_text("<|endoftext|>".join([
            "One small dog ran home.", "A little bird sings a song.",
            "The sun is bright and warm.", "Tom and Lily play together.",
            "A cat sleeps on the soft mat.", "The child likes a red ball.",
        ]), encoding="utf-8")
        prepare(source, root / "data", 4, 2, 384, 42)
        return root / "data"

    def test_future_tokens_cannot_change_earlier_logits(self):
        m, _, _ = self.modules()
        torch.manual_seed(7)
        model = m.TransformerLM(m.ModelConfig(32, 16, 2, 1, 8)).eval()
        a = torch.tensor([[1, 2, 3, 4, 5, 6]])
        b = torch.tensor([[1, 2, 3, 20, 21, 22]])
        with torch.no_grad():
            first, second = model(a), model(b)
        torch.testing.assert_close(first[:, :3], second[:, :3], atol=1e-7, rtol=1e-6)
        self.assertFalse(torch.allclose(first[:, -1], second[:, -1]))

    def test_model_can_learn_repeating_next_token_pattern(self):
        m, _, _ = self.modules()
        torch.manual_seed(3)
        model = m.TransformerLM(m.ModelConfig(8, 16, 2, 1, 8))
        x = torch.tensor([[1, 2, 3, 1, 2, 3, 1, 2]])
        y = torch.tensor([[2, 3, 1, 2, 3, 1, 2, 3]])
        opt = torch.optim.AdamW(model.parameters(), lr=0.02)
        initial = F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
        for _ in range(30):
            opt.zero_grad()
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
            loss.backward()
            opt.step()
        final = F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
        self.assertLess(final, initial * 0.3)

    def test_batch_targets_are_exactly_the_next_token(self):
        _, train, _ = self.modules()
        x, y = train.sample_batch(torch.arange(60), 3, 8, torch.Generator().manual_seed(1), "cpu")
        self.assertEqual(tuple(x.shape), (3, 8))
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
        self.assertTrue(torch.equal(y[:, -1], x[:, -1] + 1))

    def test_short_training_checkpoint_reload_generation_and_tokenizer_guard(self):
        _, train, gen = self.modules()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = self.data(root)
            out = root / "run"
            args = train.build_parser().parse_args([
                "--data-dir", str(data_dir), "--out", str(out), "--steps", "3",
                "--d-model", "16", "--heads", "2", "--layers", "1",
                "--context-length", "8", "--batch-size", "2", "--threads", "1",
                "--eval-every", "2", "--eval-batches", "2", "--log-every", "1",
                "--warmup-steps", "1",
            ])
            summary = train.run_training(args)
            self.assertEqual(summary["steps"], 3)
            self.assertTrue((out / "loss.png").exists())
            self.assertTrue((out / "last.pt").exists())
            model, tokenizer, saved = gen.load_model(out / "best.pt", data_dir, "cpu")
            self.assertGreater(saved["step"], 0)
            self.assertEqual(saved["config"]["context_length"], 8)
            last, _, _ = gen.load_model(out / "last.pt", data_dir, "cpu")
            probe = torch.tensor([[1, 2, 3]])
            again, _, _ = gen.load_model(out / "last.pt", data_dir, "cpu")
            with torch.no_grad():
                torch.testing.assert_close(last(probe), again(probe), atol=0, rtol=0)
            ids = tokenizer.encode("One").ids
            generated = gen.generate_ids(model, ids, 12, 0.8, 10, None, 42)
            self.assertEqual(generated[:len(ids)], ids)
            self.assertEqual(len(generated), len(ids) + 12)
            self.assertTrue(all(0 <= value < saved["config"]["vocab_size"] for value in generated))
            self.assertEqual(generated, gen.generate_ids(model, ids, 12, 0.8, 10, None, 42))
            with self.assertRaises(FileExistsError):
                train.run_training(args)
            with (data_dir / "tokenizer.json").open("a", encoding="utf-8") as file:
                file.write("\n")
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                gen.load_model(out / "best.pt", data_dir, "cpu")

    def test_corrupt_training_binary_is_rejected_before_training(self):
        _, train, _ = self.modules()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = self.data(root)
            with (data_dir / "train.bin").open("ab") as file:
                file.write(b"\x01\x00\x00\x00")
            args = train.build_parser().parse_args([
                "--data-dir", str(data_dir), "--out", str(root / "run"),
                "--context-length", "8", "--steps", "1",
            ])
            with self.assertRaisesRegex(ValueError, "hash"):
                train.run_training(args)
            self.assertFalse((root / "run").exists())

    def test_same_size_but_different_tokenizer_cannot_reinterpret_binary(self):
        _, train, _ = self.modules()
        with tempfile.TemporaryDirectory() as directory:
            data_dir = self.data(Path(directory))
            path = data_dir / "tokenizer.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            vocab = config["model"]["vocab"]
            vocab["A"], vocab["B"] = vocab["B"], vocab["A"]
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tokenizer|text"):
                train.load_data(data_dir, 8)


if __name__ == "__main__":
    unittest.main()
