from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from face_yolo_identity.known_db import save_db
from face_yolo_identity.profile_store import ProfileStore


class ProfileStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = ProfileStore(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_profile_crud_and_duplicate_names(self) -> None:
        profile = self.store.create("Alex Rivera", "webcam")
        self.assertEqual(profile.display_name, "Alex Rivera")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.store.create("alex rivera", "cctv")
        updated = self.store.update(profile.id, display_name="Alex R.", distance_profile="dedicated")
        self.assertEqual(updated.distance_profile, "dedicated")
        self.store.delete(profile.id)
        with self.assertRaises(KeyError):
            self.store.get(profile.id)

    def test_export_import_round_trip_uses_safe_numeric_arrays(self) -> None:
        profile = self.store.create("Morgan", "cctv")
        features = self.store.profile_dir(profile.id) / "features" / "face.npz"
        save_db(features, ["Morgan"], [np.ones(512, dtype=np.float32)])
        self.store.set_enrollment(profile.id, {"front": 2}, ready=False)
        destination = Path(self.temp.name) / "morgan.faceprofile.zip"
        self.store.export(profile.id, "inference", destination)
        with destination.open("rb") as source:
            imported = self.store.import_archive(source)
        self.assertNotEqual(imported.id, profile.id)
        self.assertEqual(imported.display_name, "Morgan (imported)")
        imported_db = self.store.profile_dir(imported.id) / "features" / "face.npz"
        with np.load(imported_db, allow_pickle=False) as data:
            self.assertEqual(data["embeddings"].shape, (1, 512))

    def test_import_rejects_path_traversal(self) -> None:
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("manifest.json", json.dumps({"schemaVersion": 1, "profile": {}}))
            archive.writestr("../escape.txt", "unsafe")
        payload.seek(0)
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            self.store.import_archive(payload)


if __name__ == "__main__":
    unittest.main()
