import { ArrowLeft, CheckCircle2, Hand, ShieldCheck, Square, Volume2, VolumeX } from "lucide-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import { BucketProgress } from "../components/BucketProgress";
import { VideoStage } from "../components/VideoStage";
import { useSessionStream } from "../hooks/useSessionStream";
import { BUCKETS, type Bucket, type CameraSession } from "../types";

const emptyCounts = Object.fromEntries(BUCKETS.map((bucket) => [bucket, 0])) as Record<Bucket, number>;

export function EnrollmentPage() {
  const { profileId = "" } = useParams();
  const queryClient = useQueryClient();
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });
  const cameras = useQuery({ queryKey: ["cameras"], queryFn: api.cameras });
  const profile = profiles.data?.find((item) => item.id === profileId);
  const [camera, setCamera] = useState(0);
  const [target, setTarget] = useState(6);
  const [consent, setConsent] = useState(false);
  const [clearExisting, setClearExisting] = useState(false);
  const [muted, setMuted] = useState(false);
  const [session, setSession] = useState<CameraSession | null>(null);
  const stream = useSessionStream(session?.id ?? null);
  const event = stream.event;
  const counts = useMemo(() => event && "bucketCounts" in event ? event.bucketCounts : profile?.bucketCounts ?? emptyCounts, [event, profile]);
  const activeBucket = event?.type === "enrollment.quality" ? event.bucket : null;
  const quality = event?.type === "enrollment.quality" ? event : null;
  const sessionStatus = event?.type === "session.status" ? event.status : session ? (stream.connected ? "running" : session.status) : "idle";
  const completed = event?.type === "enrollment.completed";
  const start = useMutation({ mutationFn: () => api.startEnrollment(profileId, camera, target, clearExisting), onSuccess: setSession });
  const stop = async () => { if (session) await api.stopSession(session.id); setSession(null); queryClient.invalidateQueries({ queryKey: ["profiles"] }); };
  const command = (action: string) => session && api.command(session.id, action);

  if (!profile && profiles.isLoading) return <p className="text-zinc-500">Loading profile…</p>;
  if (!profile) return <div><p>Profile not found.</p><Link className="button-secondary mt-4" to="/profiles">Back to profiles</Link></div>;
  return <div>
    <Link className="mb-7 inline-flex items-center gap-2 text-sm text-zinc-500 hover:text-zinc-200" to="/profiles"><ArrowLeft size={15} />Profiles</Link>
    <div className="mb-8 flex flex-wrap items-end justify-between gap-5"><div><p className="eyebrow">Guided enrollment</p><h1 className="mt-2 text-3xl font-semibold">Enroll {profile.displayName}</h1><p className="mt-2 text-sm text-zinc-500">{profile.distanceProfile} profile · balanced face and body evidence</p></div>{session && <div className={`rounded-full border px-3 py-1.5 text-xs ${stream.connected ? "border-mint/20 text-mint" : "border-amber/20 text-amber"}`}>{stream.connected ? "Live stream connected" : "Connecting to local models"}</div>}</div>
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1.7fr)_420px]">
      <div><VideoStage frameUrl={stream.frameUrl} status={completed ? "completed" : sessionStatus} detail={event && "detail" in event ? event.detail : undefined} />{session && !completed && <div className="mt-4 flex flex-wrap gap-2"><button className="button-secondary" onClick={() => command("force_save")}><Hand size={15} />Force save</button><button className="button-secondary" onClick={() => { const next = !muted; setMuted(next); command(next ? "mute" : "unmute"); }}>{muted ? <VolumeX size={15} /> : <Volume2 size={15} />}{muted ? "Unmute" : "Mute"}</button><button className="button-danger ml-auto" onClick={stop}><Square size={14} />Stop</button></div>}</div>
      <aside className="space-y-4">
        {completed ? <div className="rounded-2xl border border-mint/20 bg-mint/[0.05] p-6"><CheckCircle2 className="text-mint" size={28} /><h2 className="mt-4 text-xl font-semibold">Enrollment complete</h2><p className="mt-2 text-sm leading-6 text-zinc-400">Features are built and ready for local inference or export.</p><Link className="button-primary mt-5 w-full" to={`/inference?profiles=${profile.id}`}>Run inference</Link></div> : <div className="panel-card"><h2 className="font-semibold">Session setup</h2><label className="field-label mt-5" htmlFor="camera">Camera</label><select id="camera" className="field-input" value={camera} onChange={(event) => setCamera(Number(event.target.value))} disabled={!!session}>{cameras.data?.map((item) => <option key={item.id} value={item.id}>{item.label}{item.width ? ` · ${item.width}×${item.height}` : ""}</option>)}</select><label className="field-label mt-5" htmlFor="target">Samples per angle</label><select id="target" className="field-input" value={target} onChange={(event) => setTarget(Number(event.target.value))} disabled={!!session}><option value={4}>4 · Quick demo</option><option value={6}>6 · Balanced</option><option value={8}>8 · Accuracy</option></select>{profile.enrollmentStatus !== "empty" && <label className="mt-4 flex cursor-pointer items-center gap-3 text-sm text-zinc-400"><input className="accent-emerald-400" type="checkbox" checked={clearExisting} onChange={(event) => setClearExisting(event.target.checked)} disabled={!!session} />Replace existing samples instead of continuing</label>}<label className="mt-5 flex cursor-pointer items-start gap-3 rounded-xl border border-white/5 bg-black/10 p-4"><input className="mt-1 accent-emerald-400" type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} disabled={!!session} /><span><span className="flex items-center gap-2 text-sm font-medium"><ShieldCheck size={15} className="text-mint" />Local biometric data</span><span className="mt-1 block text-xs leading-5 text-zinc-500">I understand this demo saves face and body samples on this device.</span></span></label><button className="button-primary mt-5 w-full" disabled={!consent || !!session || start.isPending} onClick={() => start.mutate()}>{start.isPending ? "Starting models…" : profile.enrollmentStatus === "empty" ? "Start enrollment" : clearExisting ? "Clear and re-enroll" : "Continue enrollment"}</button>{start.error && <p className="mt-3 text-sm text-red-300">{start.error.message}</p>}</div>}
        <div className="panel-card"><div className="mb-4 flex items-center justify-between"><h2 className="font-semibold">Capture coverage</h2><span className={`text-xs ${quality?.accepted ? "text-mint" : "text-amber"}`}>{quality?.reason ?? "Waiting to start"}</span></div><BucketProgress counts={counts} target={target} active={activeBucket} /></div>
      </aside>
    </div>
  </div>;
}
