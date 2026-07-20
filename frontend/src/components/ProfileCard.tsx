import { Camera, Download, MoreHorizontal, Pencil, ScanFace, Trash2 } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import type { IdentityProfile } from "../types";

interface ProfileCardProps {
  profile: IdentityProfile;
  onEdit: (profile: IdentityProfile) => void;
  onDelete: (profile: IdentityProfile) => void;
}

const profileLabels = { webcam: "Webcam", dedicated: "Dedicated", cctv: "CCTV" } as const;

export function ProfileCard({ profile, onEdit, onDelete }: ProfileCardProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const samples = Object.values(profile.bucketCounts).reduce((sum, count) => sum + count, 0);
  const initials = profile.displayName.split(/\s+/).slice(0, 2).map((part) => part[0]).join("").toUpperCase();
  return (
    <article className="group relative overflow-visible rounded-2xl border border-line bg-panel p-5 transition hover:-translate-y-0.5 hover:border-mint/25 hover:shadow-glow">
      <div className="flex items-start justify-between">
        <div className="grid h-12 w-12 place-items-center rounded-2xl border border-mint/15 bg-gradient-to-br from-mint/15 to-transparent font-mono text-sm text-mint">{initials}</div>
        <div className="relative"><button className="icon-button" onClick={() => setMenuOpen((value) => !value)} aria-label={`Actions for ${profile.displayName}`}><MoreHorizontal size={18} /></button>{menuOpen && <div className="absolute right-0 top-10 z-20 w-52 rounded-xl border border-line bg-[#171e1b] p-1 shadow-xl"><button className="menu-item" onClick={() => { setMenuOpen(false); onEdit(profile); }}><Pencil size={14} />Edit</button><a className="menu-item" href={`/api/profiles/${profile.id}/export?mode=inference`}><Download size={14} />Export for inference</a><a className="menu-item" href={`/api/profiles/${profile.id}/export?mode=full`} onClick={(event) => { if (!window.confirm("A full export contains biometric-like source images. Store and share it carefully. Continue?")) event.preventDefault(); }}><Download size={14} />Export full enrollment</a><button className="menu-item text-red-300" onClick={() => { setMenuOpen(false); onDelete(profile); }}><Trash2 size={14} />Delete</button></div>}</div>
      </div>
      <div className="mt-5"><div className="flex items-center gap-2"><h3 className="text-lg font-semibold">{profile.displayName}</h3><span className={`status-dot ${profile.enrollmentStatus === "ready" ? "bg-mint" : profile.enrollmentStatus === "partial" ? "bg-amber" : "bg-zinc-600"}`} /></div><p className="mt-1 text-sm text-zinc-500">{profileLabels[profile.distanceProfile]} distance profile</p></div>
      <div className="my-5 grid grid-cols-2 gap-3 border-y border-white/5 py-4"><div><span className="metric-label">Samples</span><span className="metric-value">{samples}</span></div><div><span className="metric-label">Enrollment</span><span className="metric-value capitalize">{profile.enrollmentStatus}</span></div></div>
      <div className="flex gap-2"><Link className="button-secondary flex-1" to={`/enroll/${profile.id}`}><Camera size={15} />{samples ? "Continue" : "Enroll"}</Link><Link className={`button-primary flex-1 ${profile.enrollmentStatus === "empty" ? "pointer-events-none opacity-40" : ""}`} to={`/inference?profiles=${profile.id}`}><ScanFace size={15} />Infer</Link></div>
    </article>
  );
}
