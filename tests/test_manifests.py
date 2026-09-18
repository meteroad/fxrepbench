from __future__ import annotations

import unittest

from fxrepbench.manifests import load_tracks, validate_track


class FrozenManifestTests(unittest.TestCase):
    def test_expected_tracks(self) -> None:
        tracks = load_tracks()
        self.assertEqual(set(tracks), {"counterfx-dev80", "counterfx-200"})

    def test_all_tracks_validate(self) -> None:
        for track in load_tracks().values():
            with self.subTest(track=track.key):
                report = validate_track(track)
                self.assertEqual(report["errors"], [])


if __name__ == "__main__":
    unittest.main()
