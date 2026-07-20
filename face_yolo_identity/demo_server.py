from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
import webbrowser
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .apple_runtime import runtime_info
from .profile_store import ProfileStore
from .web_sessions import SessionError, SessionManager


class ProfileCreate(BaseModel):
    display_name: str = Field(alias="displayName")
    distance_profile: Literal["webcam", "dedicated", "cctv"] = Field(default="webcam", alias="distanceProfile")


class ProfileUpdate(BaseModel):
    display_name: str | None = Field(default=None, alias="displayName")
    distance_profile: Literal["webcam", "dedicated", "cctv"] | None = Field(default=None, alias="distanceProfile")


class EnrollmentStart(BaseModel):
    profile_id: str = Field(alias="profileId")
    camera: int = 0
    bucket_target: int = Field(default=6, ge=1, le=20, alias="bucketTarget")
    clear_existing: bool = Field(default=False, alias="clearExisting")
    start_delay: float = Field(default=4.0, ge=0, le=30, alias="startDelay")


class InferenceStart(BaseModel):
    profile_ids: list[str] = Field(min_length=1, alias="profileIds")
    camera: int = 0


class SessionCommand(BaseModel):
    action: Literal["force_save", "mute", "unmute", "reset", "stop"]


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(404, "Profile or session not found")
    if isinstance(exc, SessionError):
        return HTTPException(409, str(exc))
    return HTTPException(400, str(exc))


def _camera_list() -> list[dict]:
    # Do not probe AVFoundation from FastAPI's worker thread: on macOS that can
    # trigger a camera permission prompt before the user starts a session and
    # may fail because the capture backend expects the main run loop. The UI
    # starts with index 0; additional indices can be typed into the API or
    # discovered with the existing list_cameras.py helper.
    return [
        {"id": index, "label": "Default camera" if index == 0 else f"Camera index {index}", "width": None, "height": None}
        for index in range(4)
    ]


def create_app(data_dir: str | Path | None = None) -> FastAPI:
    store = ProfileStore(data_dir)
    sessions = SessionManager(store)
    app = FastAPI(title="Face Identity Local Demo", version="0.1.0")
    app.state.store = store
    app.state.sessions = sessions
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/runtime")
    def get_runtime() -> dict:
        info = runtime_info()
        providers = [item[0] if isinstance(item, tuple) else item for item in info.face_providers]
        face_label = next((name.replace("ExecutionProvider", "") for name in providers if name != "CPUExecutionProvider"), "CPU")
        return {
            "status": "ready",
            "accelerator": info.accelerator,
            "acceleratorLabel": info.accelerator_label,
            "detectorBackend": info.yolo_device,
            "faceBackend": face_label,
            "onnxProviders": info.onnx_providers,
            "torch": {"mps": info.torch_mps_available, "cuda": info.torch_cuda_available, "hipVersion": info.torch_hip_version, "cudaVersion": info.torch_cuda_version},
            "dataDir": str(store.root),
        }

    @app.get("/api/cameras")
    def get_cameras() -> list[dict]:
        return _camera_list()

    @app.get("/api/profiles")
    def get_profiles() -> list[dict]:
        return [profile.to_api() for profile in store.list()]

    @app.post("/api/profiles", status_code=201)
    def create_profile(body: ProfileCreate) -> dict:
        try:
            return store.create(body.display_name, body.distance_profile).to_api()
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.patch("/api/profiles/{profile_id}")
    def update_profile(profile_id: str, body: ProfileUpdate) -> dict:
        try:
            return store.update(profile_id, display_name=body.display_name, distance_profile=body.distance_profile).to_api()
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.delete("/api/profiles/{profile_id}", status_code=204)
    def delete_profile(profile_id: str) -> None:
        try:
            store.delete(profile_id)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/api/profiles/{profile_id}/export")
    def export_profile(profile_id: str, mode: Literal["inference", "full"] = "inference") -> FileResponse:
        try:
            profile = store.get(profile_id)
            handle, path_string = tempfile.mkstemp(suffix=".faceprofile.zip")
            os.close(handle)
            path = store.export(profile_id, mode, Path(path_string))
            filename = f"{profile.display_name.replace(' ', '-')}-{mode}.faceprofile.zip"
            return FileResponse(path, filename=filename, media_type="application/zip", background=BackgroundTask(path.unlink, missing_ok=True))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/profiles/import", status_code=201)
    def import_profile(file: UploadFile = File(...)) -> dict:
        try:
            return store.import_archive(file.file).to_api()
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/enrollment/sessions", status_code=201)
    def start_enrollment(body: EnrollmentStart) -> dict:
        try:
            profile = store.get(body.profile_id)
            return sessions.start_enrollment(profile, camera=body.camera, bucket_target=body.bucket_target, clear_existing=body.clear_existing, start_delay=body.start_delay).to_api()
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/inference/sessions", status_code=201)
    def start_inference(body: InferenceStart) -> dict:
        try:
            profiles = [store.get(profile_id) for profile_id in body.profile_ids]
            return sessions.start_inference(profiles, camera=body.camera).to_api()
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        try:
            return sessions.get(session_id).to_api()
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/sessions/{session_id}/commands")
    def send_command(session_id: str, body: SessionCommand) -> dict:
        try:
            session = sessions.get(session_id)
            session.command(body.action)
            return session.to_api()
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.delete("/api/sessions/{session_id}")
    def stop_session(session_id: str) -> dict:
        try:
            return sessions.stop(session_id).to_api()
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.websocket("/api/sessions/{session_id}/stream")
    async def session_stream(websocket: WebSocket, session_id: str) -> None:
        try:
            session = sessions.get(session_id)
        except KeyError:
            await websocket.close(code=4404, reason="Session not found")
            return
        await websocket.accept()
        try:
            while True:
                message = await asyncio.to_thread(session.messages.get)
                if isinstance(message, bytes):
                    await websocket.send_bytes(message)
                else:
                    await websocket.send_json(message)
                if session.status in {"completed", "cancelled", "error"} and session.messages.empty():
                    break
        except WebSocketDisconnect:
            session.stop()
        finally:
            if session.running:
                session.stop()

    frontend = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if frontend.exists():
        assets = frontend / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}")
        def spa(path: str) -> FileResponse:
            requested = frontend / path
            if path and requested.is_file() and frontend in requested.resolve().parents:
                return FileResponse(requested)
            return FileResponse(frontend / "index.html")

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local face enrollment and inference web application.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not args.no_browser:
        threading_timer = __import__("threading").Timer(1.0, lambda: webbrowser.open(f"http://{args.host}:{args.port}"))
        threading_timer.daemon = True
        threading_timer.start()
    uvicorn.run("face_yolo_identity.demo_server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
