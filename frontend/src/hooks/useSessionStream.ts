import { useEffect, useState } from "react";
import type { SessionEvent } from "../types";

export function useSessionStream(sessionId: string | null) {
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const [event, setEvent] = useState<SessionEvent | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!sessionId) {
      setFrameUrl(null);
      setEvent(null);
      setConnected(false);
      return;
    }
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${window.location.host}/api/sessions/${sessionId}/stream`);
    socket.binaryType = "blob";
    let latestUrl: string | null = null;
    socket.onopen = () => setConnected(true);
    socket.onclose = () => setConnected(false);
    socket.onmessage = (message) => {
      if (message.data instanceof Blob) {
        const nextUrl = URL.createObjectURL(message.data);
        if (latestUrl) URL.revokeObjectURL(latestUrl);
        latestUrl = nextUrl;
        setFrameUrl(nextUrl);
      } else {
        setEvent(JSON.parse(message.data) as SessionEvent);
      }
    };
    return () => {
      socket.close();
      if (latestUrl) URL.revokeObjectURL(latestUrl);
    };
  }, [sessionId]);

  return { frameUrl, event, connected };
}
