export const BUCKETS = ["front", "turn_left", "turn_right", "close", "far", "back"] as const;
export type Bucket = (typeof BUCKETS)[number];
export type DistanceProfile = "webcam" | "dedicated" | "cctv";

export interface IdentityProfile {
  id: string;
  displayName: string;
  distanceProfile: DistanceProfile;
  createdAt: string;
  updatedAt: string;
  enrollmentStatus: "empty" | "partial" | "ready";
  bucketCounts: Record<Bucket, number>;
}

export interface RuntimeInfo {
  status: string;
  accelerator: string;
  acceleratorLabel: string;
  detectorBackend: string;
  faceBackend: string;
  onnxProviders: string[];
  dataDir: string;
  torch: { mps: boolean; cuda: boolean; hipVersion: string | null; cudaVersion: string | null };
}

export interface CameraInfo {
  id: number;
  label: string;
  width: number | null;
  height: number | null;
}

export interface CameraSession {
  id: string;
  kind: "enrollment" | "inference";
  camera: number;
  status: string;
  error: string | null;
}

export interface DetectionResult {
  trackId: number | null;
  name: string | null;
  score: number | null;
  source: string;
  conflict: boolean;
  box: number[];
}

export type SessionEvent =
  | { type: "session.status"; status: string; detail?: string }
  | { type: "session.error"; message: string }
  | { type: "runtime.status"; detectorBackend: string; faceBackend: string; faceProviders: string[] }
  | { type: "enrollment.quality"; accepted: boolean; reason: string; bucket: Bucket | null; bucketCounts: Record<Bucket, number>; target: number; waitingSeconds: number }
  | { type: "enrollment.progress"; bucket: Bucket; bucketCounts: Record<Bucket, number>; target: number; saved: number }
  | { type: "enrollment.completed"; profile: IdentityProfile; bucketCounts: Record<Bucket, number> }
  | { type: "inference.detections"; fps: number | null; detections: DetectionResult[] };
