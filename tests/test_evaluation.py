from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fxrepbench.evaluation import load_submission


class SubmissionTests(unittest.TestCase):
    def test_loads_relative_jsonl_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission = root / "submission.jsonl"
            submission.write_text(
                json.dumps(
                    {
                        "item_id": "example",
                        "candidate_id": "001",
                        "audio_path": "audio/001.wav",
                        "score": 0.5,
                        "chain_order": ["compressor", "reverb"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = load_submission(submission)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].audio_path, root / "audio/001.wav")
        self.assertEqual(rows[0].chain_order, ("compressor", "reverb"))

    def test_rejects_unknown_effect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            submission = Path(directory) / "submission.jsonl"
            submission.write_text(
                json.dumps(
                    {
                        "item_id": "example",
                        "audio_path": "audio.wav",
                        "chain_order": ["unknown_fx"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown effects"):
                load_submission(submission)
