import { Fingerprint, ScanFace, UsersRound } from "lucide-react";
import type { PropsWithChildren } from "react";
import { NavLink } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { RuntimeBadge } from "./RuntimeBadge";

const navItems = [
  { to: "/profiles", label: "Profiles", icon: UsersRound },
  { to: "/inference", label: "Inference", icon: ScanFace },
] as const;

export function AppShell({ children }: PropsWithChildren) {
  const runtime = useQuery({ queryKey: ["runtime"], queryFn: api.runtime, refetchInterval: 30_000 });
  return (
    <div className="min-h-screen bg-ink text-zinc-100">
      <header className="sticky top-0 z-40 border-b border-white/5 bg-ink/85 backdrop-blur-xl">
        <div className="mx-auto flex h-18 max-w-[1500px] items-center justify-between px-6 py-4 lg:px-10">
          <NavLink to="/profiles" className="flex items-center gap-3">
            <span className="grid h-10 w-10 place-items-center rounded-xl border border-mint/20 bg-mint/10 text-mint"><Fingerprint size={21} /></span>
            <span><span className="block text-sm font-semibold tracking-wide">Local Identity Lab</span><span className="block text-[11px] uppercase tracking-[0.22em] text-zinc-600">Private · on device</span></span>
          </NavLink>
          <nav className="flex items-center gap-1 rounded-xl border border-white/5 bg-white/[0.025] p-1" aria-label="Primary navigation">
            {navItems.map(({ to, label, icon: Icon }) => (
              <NavLink key={to} to={to} className={({ isActive }) => `nav-link ${isActive ? "nav-link-active" : ""}`}><Icon size={16} />{label}</NavLink>
            ))}
          </nav>
          <RuntimeBadge runtime={runtime.data} offline={runtime.isError} />
        </div>
      </header>
      {runtime.isError && (
        <div className="border-b border-red-400/15 bg-red-400/[0.06] px-6 py-2 text-center text-sm text-red-200">
          Local service unavailable. Start it with <code className="font-mono">uv run python -m face_yolo_identity.demo_server</code>.
        </div>
      )}
      <main className="mx-auto max-w-[1500px] px-6 py-10 lg:px-10">{children}</main>
    </div>
  );
}
