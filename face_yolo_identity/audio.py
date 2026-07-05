from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import threading
import time


class AudioPrompter:
    """Tiny non-blocking sound/speech helper for enrollment.

    On macOS it uses the built-in `say` command when speech is enabled. On
    other platforms it falls back to a terminal bell. Audio is intentionally
    optional: no audio failure should break enrollment.
    """

    def __init__(self, enabled: bool = True, speech: bool = True, min_interval: float = 1.2) -> None:
        self.enabled = enabled
        self.speech = speech
        self.min_interval = min_interval
        self._last_message = ""
        self._last_time = 0.0
        self._say_path = shutil.which("say") if platform.system() == "Darwin" else None

    def prompt(self, message: str, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.time()
        if not force and message == self._last_message and now - self._last_time < self.min_interval:
            return
        if not force and now - self._last_time < self.min_interval:
            return
        self._last_message = message
        self._last_time = now
        threading.Thread(target=self._play, args=(message,), daemon=True).start()

    def _play(self, message: str) -> None:
        try:
            if self.speech and self._say_path:
                subprocess.run([self._say_path, message], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4)
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            try:
                sys.stdout.write("\a")
                sys.stdout.flush()
            except Exception:
                pass
