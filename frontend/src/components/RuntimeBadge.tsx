import { Cpu, Server, X } from "lucide-react";
import { useState } from "react";
import type { RuntimeInfo } from "../types";

interface RuntimeBadgeProps {
  runtime?: RuntimeInfo;
  offline: boolean;
}

export function RuntimeBadge({ runtime, offline }: RuntimeBadgeProps) {
  const [open, setOpen] = useState(false);
  const label = offline ? "Service offline" : runtime ? `${runtime.acceleratorLabel} + ${runtime.faceBackend}` : "Checking runtime";
  return (
    <div className="relative">
      <button className={`runtime-badge ${offline ? "border-red-400/30 text-red-300" : ""}`} onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        <span className={`h-2 w-2 rounded-full ${offline ? "bg-red-400" : "bg-mint shadow-[0_0_12px_#55e6a5]"}`} />
        {label}
      </button>
      {open && (
        <div className="absolute right-0 top-12 z-50 w-[360px] rounded-2xl border border-line bg-[#121816] p-5 shadow-glow">
          <div className="mb-4 flex items-center justify-between">
            <div className="flex items-center gap-2 font-semibold"><Server size={16} /> Runtime details</div>
            <button className="icon-button" onClick={() => setOpen(false)} aria-label="Close runtime details"><X size={16} /></button>
          </div>
          {offline ? (
            <div className="space-y-3 text-sm text-zinc-400">
              <p>The local Python service is not responding.</p>
              <code className="block rounded-lg bg-black/30 p-3 text-xs text-zinc-200">uv run python -m face_yolo_identity.demo_server</code>
            </div>
          ) : runtime ? (
            <dl className="grid grid-cols-[120px_1fr] gap-y-3 text-sm">
              <dt className="text-zinc-500">Detector</dt><dd className="font-mono text-zinc-200">{runtime.detectorBackend}</dd>
              <dt className="text-zinc-500">Face engine</dt><dd className="font-mono text-zinc-200">{runtime.faceBackend}</dd>
              <dt className="text-zinc-500">Accelerator</dt><dd className="text-zinc-200">{runtime.acceleratorLabel}</dd>
              <dt className="text-zinc-500">Profile data</dt><dd className="break-all text-xs text-zinc-400">{runtime.dataDir}</dd>
            </dl>
          ) : <div className="flex items-center gap-2 text-sm text-zinc-400"><Cpu className="animate-pulse" size={16} /> Detecting local hardware…</div>}
        </div>
      )}
    </div>
  );
}
