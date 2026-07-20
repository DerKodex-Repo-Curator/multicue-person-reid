import { X } from "lucide-react";
import { useEffect, useState, type FormEvent } from "react";
import type { DistanceProfile, IdentityProfile } from "../types";

interface ProfileDialogProps {
  profile?: IdentityProfile | null;
  busy: boolean;
  error?: string;
  onClose: () => void;
  onSubmit: (displayName: string, distanceProfile: DistanceProfile) => void;
}

const distanceOptions: { value: DistanceProfile; label: string; detail: string }[] = [
  { value: "webcam", label: "Webcam", detail: "Near field · 72% / 46%" },
  { value: "dedicated", label: "Dedicated", detail: "Tripod or USB · 62% / 34%" },
  { value: "cctv", label: "CCTV", detail: "Distant framing · 42% / 18%" },
];

export function ProfileDialog({ profile, busy, error, onClose, onSubmit }: ProfileDialogProps) {
  const [name, setName] = useState(profile?.displayName ?? "");
  const [distance, setDistance] = useState<DistanceProfile>(profile?.distanceProfile ?? "webcam");
  useEffect(() => {
    setName(profile?.displayName ?? "");
    setDistance(profile?.distanceProfile ?? "webcam");
  }, [profile]);
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (name.trim()) onSubmit(name.trim(), distance);
  };
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/70 p-4 backdrop-blur-sm" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <form className="w-full max-w-lg rounded-3xl border border-line bg-panel p-7 shadow-glow" onSubmit={submit}>
        <div className="mb-7 flex items-start justify-between"><div><p className="eyebrow">Identity profile</p><h2 className="mt-2 text-2xl font-semibold">{profile ? "Edit profile" : "Create a profile"}</h2></div><button type="button" className="icon-button" onClick={onClose} aria-label="Close"><X size={18} /></button></div>
        <label className="field-label" htmlFor="profile-name">Display name</label>
        <input id="profile-name" className="field-input" value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. Alex Rivera" autoFocus maxLength={80} />
        <fieldset className="mt-6"><legend className="field-label">Camera distance profile</legend><div className="mt-2 grid gap-2">
          {distanceOptions.map((option) => <label key={option.value} className={`choice-row ${distance === option.value ? "choice-row-active" : ""}`}><input type="radio" className="sr-only" name="distance" value={option.value} checked={distance === option.value} onChange={() => setDistance(option.value)} /><span><span className="block text-sm font-medium">{option.label}</span><span className="text-xs text-zinc-500">{option.detail}</span></span><span className={`h-3 w-3 rounded-full border ${distance === option.value ? "border-mint bg-mint" : "border-zinc-600"}`} /></label>)}
        </div></fieldset>
        {error && <p className="mt-4 text-sm text-red-300">{error}</p>}
        <div className="mt-7 flex justify-end gap-3"><button type="button" className="button-secondary" onClick={onClose}>Cancel</button><button className="button-primary" disabled={busy || !name.trim()}>{busy ? "Saving…" : profile ? "Save changes" : "Create profile"}</button></div>
      </form>
    </div>
  );
}
