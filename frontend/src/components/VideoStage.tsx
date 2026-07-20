import { CameraOff, LoaderCircle } from "lucide-react";

interface VideoStageProps {
  frameUrl: string | null;
  status: string;
  detail?: string;
}

export function VideoStage({ frameUrl, status, detail }: VideoStageProps) {
  return <div className="relative aspect-video overflow-hidden rounded-2xl border border-line bg-[#050807] shadow-glow">
    {frameUrl ? <img className="h-full w-full object-contain" src={frameUrl} alt="Live annotated camera feed" /> : <div className="absolute inset-0 grid place-items-center"><div className="text-center text-zinc-600">{status === "idle" ? <CameraOff className="mx-auto mb-3" size={30} /> : <LoaderCircle className="mx-auto mb-3 animate-spin text-mint" size={30} />}<p className="text-sm">{status === "idle" ? "Camera is idle" : detail ?? "Preparing local models…"}</p></div></div>}
    <div className="absolute left-4 top-4 flex items-center gap-2 rounded-full border border-white/10 bg-black/65 px-3 py-1.5 text-[11px] font-medium uppercase tracking-[0.16em] backdrop-blur"><span className={`h-1.5 w-1.5 rounded-full ${status === "running" ? "bg-red-400 animate-pulse" : "bg-zinc-500"}`} />{status}</div>
  </div>;
}
