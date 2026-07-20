import { FileUp, Plus, ShieldCheck, Sparkles } from "lucide-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { api } from "../api";
import { ProfileCard } from "../components/ProfileCard";
import { ProfileDialog } from "../components/ProfileDialog";
import type { DistanceProfile, IdentityProfile } from "../types";

export function ProfilesPage() {
  const queryClient = useQueryClient();
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });
  const [dialog, setDialog] = useState<"create" | IdentityProfile | null>(null);
  const [formError, setFormError] = useState<string>();
  const importInput = useRef<HTMLInputElement>(null);
  const save = useMutation({
    mutationFn: ({ name, distance }: { name: string; distance: DistanceProfile }) =>
      dialog && dialog !== "create" ? api.updateProfile(dialog.id, name, distance) : api.createProfile(name, distance),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["profiles"] }); setDialog(null); setFormError(undefined); },
    onError: (error) => setFormError(error.message),
  });
  const remove = useMutation({ mutationFn: api.deleteProfile, onSuccess: () => queryClient.invalidateQueries({ queryKey: ["profiles"] }) });
  const importProfile = useMutation({ mutationFn: api.importProfile, onSuccess: () => queryClient.invalidateQueries({ queryKey: ["profiles"] }) });

  const handleDelete = (profile: IdentityProfile) => {
    if (window.confirm(`Delete ${profile.displayName} and all locally stored enrollment data? This cannot be undone.`)) remove.mutate(profile.id);
  };
  const handleImport = (file?: File) => { if (file) importProfile.mutate(file); };

  return <>
    <section className="grid gap-8 border-b border-white/5 pb-10 lg:grid-cols-[1fr_420px] lg:items-end">
      <div><p className="eyebrow">Local profile vault</p><h1 className="mt-3 max-w-3xl text-4xl font-semibold tracking-[-0.035em] sm:text-5xl">Enroll once. Recognize locally. Move profiles safely.</h1><p className="mt-5 max-w-2xl text-base leading-7 text-zinc-400">Each identity keeps its own face and body enrollment, tuned for the camera distance where it will be used.</p></div>
      <div className="rounded-2xl border border-mint/15 bg-gradient-to-br from-mint/[0.07] to-transparent p-5"><div className="flex items-center gap-3 text-mint"><ShieldCheck size={19} /><span className="text-sm font-medium">Private by default</span></div><p className="mt-3 text-sm leading-6 text-zinc-400">Images and embeddings stay on this device. Nothing is uploaded or synchronized.</p></div>
    </section>
    <section className="pt-9">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4"><div><h2 className="text-xl font-semibold">Identity profiles</h2><p className="mt-1 text-sm text-zinc-500">{profiles.data?.length ?? 0} stored locally</p></div><div className="flex gap-2"><input ref={importInput} className="hidden" type="file" accept=".zip,.faceprofile.zip" onChange={(event) => handleImport(event.target.files?.[0])} /><button className="button-secondary" onClick={() => importInput.current?.click()} disabled={importProfile.isPending}><FileUp size={16} />{importProfile.isPending ? "Importing…" : "Import"}</button><button className="button-primary" onClick={() => setDialog("create")}><Plus size={16} />New profile</button></div></div>
      {profiles.isLoading ? <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">{[0,1,2].map((item) => <div key={item} className="h-72 animate-pulse rounded-2xl border border-white/5 bg-white/[0.025]" />)}</div> : profiles.data?.length ? <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">{profiles.data.map((profile) => <ProfileCard key={profile.id} profile={profile} onEdit={setDialog} onDelete={handleDelete} />)}</div> : <div className="grid min-h-80 place-items-center rounded-3xl border border-dashed border-line bg-white/[0.015] text-center"><div><span className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-mint/10 text-mint"><Sparkles size={20} /></span><h3 className="mt-4 font-semibold">No identities enrolled yet</h3><p className="mt-2 max-w-sm text-sm text-zinc-500">Create a profile, choose its camera distance, then follow the guided capture.</p><button className="button-primary mx-auto mt-5" onClick={() => setDialog("create")}><Plus size={16} />Create first profile</button></div></div>}
      {(profiles.error || importProfile.error || remove.error) && <p className="mt-4 text-sm text-red-300">{(profiles.error ?? importProfile.error ?? remove.error)?.message}</p>}
    </section>
    {dialog && <ProfileDialog profile={dialog === "create" ? null : dialog} busy={save.isPending} error={formError} onClose={() => { setDialog(null); setFormError(undefined); }} onSubmit={(name, distance) => save.mutate({ name, distance })} />}
  </>;
}
