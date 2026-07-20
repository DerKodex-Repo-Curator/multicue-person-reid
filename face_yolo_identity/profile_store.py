from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Literal

try:
    from platformdirs import user_data_path
except ImportError:  # Keep profile management usable before optional web deps are synced.
    def user_data_path(appname: str, appauthor: str) -> Path:
        if os.name == "nt":
            return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / appname
        if __import__("platform").system() == "Darwin":
            return Path.home() / "Library" / "Application Support" / appname
        return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / appname


DistanceProfile = Literal["webcam", "dedicated", "cctv"]
EnrollmentStatus = Literal["empty", "partial", "ready"]
BUCKETS = ("front", "turn_left", "turn_right", "close", "far", "back")
SCHEMA_VERSION = 1
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_FILES = 2_000
ALLOWED_ARCHIVE_SUFFIXES = {".jpg", ".jpeg", ".png", ".jsonl", ".npz"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_data_dir() -> Path:
    configured = os.environ.get("FACE_ID_DATA_DIR")
    return Path(configured).expanduser() if configured else user_data_path("face-yolo-identity", "Codex")


@dataclass
class IdentityProfile:
    id: str
    display_name: str
    distance_profile: DistanceProfile
    created_at: str
    updated_at: str
    enrollment_status: EnrollmentStatus = "empty"
    bucket_counts: dict[str, int] = field(default_factory=lambda: {bucket: 0 for bucket in BUCKETS})

    def to_api(self) -> dict:
        data = asdict(self)
        return {
            "id": data["id"],
            "displayName": data["display_name"],
            "distanceProfile": data["distance_profile"],
            "createdAt": data["created_at"],
            "updatedAt": data["updated_at"],
            "enrollmentStatus": data["enrollment_status"],
            "bucketCounts": data["bucket_counts"],
        }

    @classmethod
    def from_json(cls, data: dict) -> "IdentityProfile":
        return cls(
            id=str(data["id"]),
            display_name=str(data.get("display_name", data.get("displayName", ""))),
            distance_profile=data.get("distance_profile", data.get("distanceProfile", "webcam")),
            created_at=str(data.get("created_at", data.get("createdAt", _now()))),
            updated_at=str(data.get("updated_at", data.get("updatedAt", _now()))),
            enrollment_status=data.get("enrollment_status", data.get("enrollmentStatus", "empty")),
            bucket_counts={bucket: int(data.get("bucket_counts", data.get("bucketCounts", {})).get(bucket, 0)) for bucket in BUCKETS},
        )


class ProfileStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else default_data_dir()
        self.profiles_dir = self.root / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def profile_dir(self, profile_id: str) -> Path:
        self._validate_id(profile_id)
        return self.profiles_dir / profile_id

    @staticmethod
    def _validate_id(profile_id: str) -> None:
        try:
            parsed = uuid.UUID(profile_id)
        except ValueError as exc:
            raise ValueError("Invalid profile ID") from exc
        if str(parsed) != profile_id:
            raise ValueError("Invalid profile ID")

    @staticmethod
    def _validate_name(display_name: str) -> str:
        name = " ".join(display_name.strip().split())
        if not 1 <= len(name) <= 80:
            raise ValueError("Display name must contain 1 to 80 characters")
        return name

    @staticmethod
    def _validate_distance(value: str) -> DistanceProfile:
        if value not in {"webcam", "dedicated", "cctv"}:
            raise ValueError("Distance profile must be webcam, dedicated, or cctv")
        return value  # type: ignore[return-value]

    def _write(self, profile: IdentityProfile) -> None:
        directory = self.profile_dir(profile.id)
        (directory / "samples" / "face").mkdir(parents=True, exist_ok=True)
        (directory / "samples" / "body").mkdir(parents=True, exist_ok=True)
        (directory / "samples" / "unified_frames").mkdir(parents=True, exist_ok=True)
        (directory / "features").mkdir(parents=True, exist_ok=True)
        target = directory / "profile.json"
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")
        temp.replace(target)

    def list(self) -> list[IdentityProfile]:
        profiles: list[IdentityProfile] = []
        for path in self.profiles_dir.glob("*/profile.json"):
            try:
                profiles.append(IdentityProfile.from_json(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return sorted(profiles, key=lambda item: item.updated_at, reverse=True)

    def get(self, profile_id: str) -> IdentityProfile:
        path = self.profile_dir(profile_id) / "profile.json"
        if not path.exists():
            raise KeyError(profile_id)
        return IdentityProfile.from_json(json.loads(path.read_text(encoding="utf-8")))

    def create(self, display_name: str, distance_profile: str) -> IdentityProfile:
        name = self._validate_name(display_name)
        if any(item.display_name.casefold() == name.casefold() for item in self.list()):
            raise ValueError("A profile with this display name already exists")
        now = _now()
        profile = IdentityProfile(str(uuid.uuid4()), name, self._validate_distance(distance_profile), now, now)
        self._write(profile)
        return profile

    def update(self, profile_id: str, *, display_name: str | None = None, distance_profile: str | None = None) -> IdentityProfile:
        profile = self.get(profile_id)
        if display_name is not None:
            name = self._validate_name(display_name)
            if any(item.id != profile_id and item.display_name.casefold() == name.casefold() for item in self.list()):
                raise ValueError("A profile with this display name already exists")
            profile.display_name = name
        if distance_profile is not None:
            profile.distance_profile = self._validate_distance(distance_profile)
        profile.updated_at = _now()
        self._write(profile)
        return profile

    def set_enrollment(self, profile_id: str, counts: dict[str, int], *, ready: bool) -> IdentityProfile:
        profile = self.get(profile_id)
        profile.bucket_counts = {bucket: max(0, int(counts.get(bucket, 0))) for bucket in BUCKETS}
        profile.enrollment_status = "ready" if ready else "partial" if sum(profile.bucket_counts.values()) else "empty"
        profile.updated_at = _now()
        self._write(profile)
        return profile

    def delete(self, profile_id: str) -> None:
        directory = self.profile_dir(profile_id)
        if not directory.exists():
            raise KeyError(profile_id)
        shutil.rmtree(directory)

    @staticmethod
    def _hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def export(self, profile_id: str, mode: Literal["inference", "full"], destination: Path) -> Path:
        profile = self.get(profile_id)
        source = self.profile_dir(profile_id)
        candidates = [path for path in (source / "features").rglob("*") if path.is_file()]
        if mode == "full":
            candidates += [path for path in (source / "samples").rglob("*") if path.is_file()]
        files = []
        for path in candidates:
            relative = path.relative_to(source).as_posix()
            files.append({"path": relative, "size": path.stat().st_size, "sha256": self._hash(path)})
        manifest = {"schemaVersion": SCHEMA_VERSION, "mode": mode, "profile": profile.to_api(), "files": files}
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))
            for item in files:
                archive.write(source / item["path"], item["path"])
        return destination

    def import_archive(self, file: BinaryIO) -> IdentityProfile:
        with tempfile.NamedTemporaryFile(suffix=".faceprofile.zip") as temp:
            shutil.copyfileobj(file, temp)
            temp.flush()
            if Path(temp.name).stat().st_size > MAX_ARCHIVE_BYTES:
                raise ValueError("Profile archive is too large")
            with zipfile.ZipFile(temp.name) as archive:
                infos = archive.infolist()
                if len(infos) > MAX_ARCHIVE_FILES:
                    raise ValueError("Profile archive contains too many files")
                if sum(info.file_size for info in infos) > MAX_ARCHIVE_BYTES:
                    raise ValueError("Profile archive expands beyond the allowed size")
                for info in infos:
                    path = Path(info.filename)
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError("Profile archive contains an unsafe path")
                    if info.filename != "manifest.json" and path.suffix.lower() not in ALLOWED_ARCHIVE_SUFFIXES:
                        raise ValueError("Profile archive contains an unsupported file type")
                try:
                    manifest = json.loads(archive.read("manifest.json"))
                except (KeyError, json.JSONDecodeError) as exc:
                    raise ValueError("Profile archive has no valid manifest") from exc
                if manifest.get("schemaVersion") != SCHEMA_VERSION:
                    raise ValueError("Unsupported profile archive version")
                original = IdentityProfile.from_json(manifest["profile"])
                new_id = str(uuid.uuid4())
                name = original.display_name
                existing = {item.display_name.casefold() for item in self.list()}
                if name.casefold() in existing:
                    name = f"{name} (imported)"
                profile = IdentityProfile(new_id, self._validate_name(name), self._validate_distance(original.distance_profile), _now(), _now(), original.enrollment_status, original.bucket_counts)
                target = self.profile_dir(new_id)
                try:
                    for item in manifest.get("files", []):
                        relative = Path(str(item["path"]))
                        if relative.parts[0] not in {"samples", "features"}:
                            raise ValueError("Profile archive contains an unsupported file")
                        payload = archive.read(relative.as_posix())
                        if len(payload) != int(item["size"]) or hashlib.sha256(payload).hexdigest() != item["sha256"]:
                            raise ValueError("Profile archive checksum mismatch")
                        output = target / relative
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_bytes(payload)
                        if output.suffix.lower() == ".npz":
                            # Access every array so object/pickle payloads and corrupt
                            # archives fail during import rather than inference.
                            import numpy as np

                            with np.load(output, allow_pickle=False) as arrays:
                                for key in arrays.files:
                                    np.asarray(arrays[key])
                    self._write(profile)
                except Exception:
                    shutil.rmtree(target, ignore_errors=True)
                    raise
                return profile
