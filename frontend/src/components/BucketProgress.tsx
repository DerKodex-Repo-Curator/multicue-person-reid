import type { Bucket } from "../types";
import { BUCKETS } from "../types";

interface BucketProgressProps {
  counts: Record<Bucket, number>;
  target: number;
  active?: Bucket | null;
}

const labels: Record<Bucket, string> = { front: "Front", turn_left: "Turn left", turn_right: "Turn right", close: "Close", far: "Far", back: "Back" };

export function BucketProgress({ counts, target, active }: BucketProgressProps) {
  return <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{BUCKETS.map((bucket) => {
    const complete = counts[bucket] >= target;
    return <div key={bucket} className={`rounded-xl border p-3 ${active === bucket ? "border-amber/40 bg-amber/[0.04]" : "border-white/5 bg-black/10"}`}><div className="mb-2 flex items-center justify-between text-xs"><span className={complete ? "text-mint" : "text-zinc-400"}>{labels[bucket]}</span><span className="font-mono text-zinc-500">{counts[bucket]}/{target}</span></div><div className="h-1.5 overflow-hidden rounded-full bg-white/5"><div className={`h-full rounded-full transition-all ${complete ? "bg-mint" : "bg-amber"}`} style={{ width: `${Math.min(100, (counts[bucket] / target) * 100)}%` }} /></div></div>;
  })}</div>;
}
