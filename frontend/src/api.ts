import type { CameraInfo, CameraSession, DistanceProfile, IdentityProfile, RuntimeInfo } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: init?.body instanceof FormData ? init.headers : { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? "Request failed");
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  runtime: () => request<RuntimeInfo>("/api/runtime"),
  cameras: () => request<CameraInfo[]>("/api/cameras"),
  profiles: () => request<IdentityProfile[]>("/api/profiles"),
  createProfile: (displayName: string, distanceProfile: DistanceProfile) =>
    request<IdentityProfile>("/api/profiles", { method: "POST", body: JSON.stringify({ displayName, distanceProfile }) }),
  updateProfile: (id: string, displayName: string, distanceProfile: DistanceProfile) =>
    request<IdentityProfile>(`/api/profiles/${id}`, { method: "PATCH", body: JSON.stringify({ displayName, distanceProfile }) }),
  deleteProfile: (id: string) => request<void>(`/api/profiles/${id}`, { method: "DELETE" }),
  importProfile: (file: File) => {
    const data = new FormData();
    data.append("file", file);
    return request<IdentityProfile>("/api/profiles/import", { method: "POST", body: data });
  },
  startEnrollment: (profileId: string, camera: number, bucketTarget: number, clearExisting: boolean) =>
    request<CameraSession>("/api/enrollment/sessions", { method: "POST", body: JSON.stringify({ profileId, camera, bucketTarget, clearExisting, startDelay: 4 }) }),
  startInference: (profileIds: string[], camera: number) =>
    request<CameraSession>("/api/inference/sessions", { method: "POST", body: JSON.stringify({ profileIds, camera }) }),
  command: (sessionId: string, action: string) =>
    request<CameraSession>(`/api/sessions/${sessionId}/commands`, { method: "POST", body: JSON.stringify({ action }) }),
  stopSession: (sessionId: string) => request<CameraSession>(`/api/sessions/${sessionId}`, { method: "DELETE" }),
};
