import { Activity, RotateCcw, ScanFace, Square } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api";
import { VideoStage } from "../components/VideoStage";
import { useSessionStream } from "../hooks/useSessionStream";
import type { CameraSession, DetectionResult } from "../types";

export function InferencePage() {
  const [params] = useSearchParams();
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });
  const cameras = useQuery({ queryKey: ["cameras"], queryFn: api.cameras });
  const [selected, setSelected] = useState<string[]>([]);
  const [camera, setCamera] = useState(0);
  const [session, setSession] = useState<CameraSession | null>(null);
  const stream = useSessionStream(session?.id ?? null);
  useEffect(() => { const initial = params.get("profiles")?.split(",").filter(Boolean) ?? []; if (initial.length) setSelected(initial); }, [params]);
  const readyProfiles = profiles.data?.filter((profile) => profile.enrollmentStatus !== "empty") ?? [];
  const inferenceEvent = stream.event?.type === "inference.detections" ? stream.event : null;
  const detections: DetectionResult[] = inferenceEvent?.detections ?? [];
  const identified = useMemo(() => detections.filter((item) => item.name), [detections]);
  const status = stream.event?.type === "session.status" ? stream.event.status : session ? (stream.connected ? "running" : session.status) : "idle";
  const start = useMutation({ mutationFn: () => api.startInference(selected, camera), onSuccess: setSession });
  const stop = async () => { if (session) await api.stopSession(session.id); setSession(null); };
  const toggleProfile = (id: string) => setSelected((items) => items.includes(id) ? items.filter((item) => item !== id) : [...items, id]);
  return <div>
    <div className="mb-8 flex flex-wrap items-end justify-between gap-5"><div><p className="eyebrow">Live inference</p><h1 className="mt-2 text-3xl font-semibold">Recognize enrolled identities</h1><p className="mt-2 text-sm text-zinc-500">Face evidence leads. Weak or conflicting matches remain unknown.</p></div><div className="flex items-center gap-2 rounded-full border border-white/5 bg-white/[0.025] px-4 py-2 text-xs text-zinc-400"><Activity size={14} className={session ? "text-mint" : ""} />{inferenceEvent?.fps ? `${inferenceEvent.fps} FPS` : "Engine idle"}</div></div>
    <div className="grid gap-6 xl:grid-cols-[340px_minmax(0,1fr)]">
      <aside className="space-y-4"><div className="panel-card"><h2 className="font-semibold">Recognition set</h2><p className="mt-1 text-xs leading-5 text-zinc-500">Choose the profiles that may be named in this session.</p><div className="mt-4 max-h-72 space-y-2 overflow-auto">{readyProfiles.length ? readyProfiles.map((profile) => <label key={profile.id} className={`choice-row ${selected.includes(profile.id) ? "choice-row-active" : ""}`}><input type="checkbox" className="accent-emerald-400" checked={selected.includes(profile.id)} onChange={() => toggleProfile(profile.id)} disabled={!!session} /><span className="min-w-0"><span className="block truncate text-sm font-medium">{profile.displayName}</span><span className="text-xs capitalize text-zinc-500">{profile.distanceProfile} · {profile.enrollmentStatus}</span></span></label>) : <p className="rounded-xl border border-dashed border-line p-4 text-sm text-zinc-500">Enroll a profile before starting inference.</p>}</div><label className="field-label mt-5" htmlFor="inference-camera">Camera</label><select id="inference-camera" className="field-input" value={camera} onChange={(event) => setCamera(Number(event.target.value))} disabled={!!session}>{cameras.data?.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select>{session ? <button className="button-danger mt-5 w-full" onClick={stop}><Square size={14} />Stop inference</button> : <button className="button-primary mt-5 w-full" onClick={() => start.mutate()} disabled={!selected.length || start.isPending}><ScanFace size={16} />{start.isPending ? "Loading models…" : "Start inference"}</button>}{start.error && <p className="mt-3 text-sm text-red-300">{start.error.message}</p>}</div>
        <div className="panel-card"><div className="flex items-center justify-between"><h2 className="font-semibold">Current evidence</h2>{session && <button className="icon-button" onClick={() => api.command(session.id, "reset")} aria-label="Reset evidence"><RotateCcw size={15} /></button>}</div><div className="mt-4 space-y-2">{detections.length ? detections.map((detection) => <div key={detection.trackId ?? detection.box.join(":")} className="rounded-xl border border-white/5 bg-black/10 p-3"><div className="flex items-center justify-between"><span className={`text-sm font-medium ${detection.conflict ? "text-red-300" : detection.name ? "text-mint" : "text-amber"}`}>{detection.conflict ? "Conflict" : detection.name ?? "Unknown"}</span><span className="font-mono text-xs text-zinc-500">{detection.score?.toFixed(2) ?? "—"}</span></div><p className="mt-1 text-xs text-zinc-600">{detection.source}</p></div>) : <p className="py-5 text-center text-sm text-zinc-600">No identity evidence yet</p>}</div>{identified.length > 0 && <p className="mt-3 text-xs text-zinc-500">{identified.length} known {identified.length === 1 ? "identity" : "identities"} visible</p>}</div></aside>
      <VideoStage frameUrl={stream.frameUrl} status={status} detail={stream.event && "detail" in stream.event ? stream.event.detail : undefined} />
    </div>
  </div>;
}
