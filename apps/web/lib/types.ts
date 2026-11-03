export interface MemoryEvent {
  event_id: string;
  camera_id: string;
  zone_id: string;
  track_id: string;
  object_identity_id?: string | null;
  class_name: string;
  confidence: number;
  timestamp_utc: string;
  bbox_xyxy: [number, number, number, number];
  embedding_id?: string | null;
  snapshot_uri: string;
  crop_uri?: string | null;
  event_label?: string;
}

export interface QueryRequest {
  question: string;
  camera_id?: string;
  zone_id?: string;
  require_evidence?: boolean;
}

export interface QueryResponse {
  question: string;
  answer: string;
  object_class?: string | null;
  camera_id?: string | null;
  zone_id?: string | null;
  timestamp_utc?: string | null;
  time_ago?: string | null;
  confidence?: number | null;
  event_id?: string | null;
  snapshot_uri?: string | null;
  crop_uri?: string | null;
  is_uncertain?: boolean;
  disclaimer?: string | null;
  sightings_count?: number | null;
  all_known_objects?: string[] | null;
}

export interface TimelineEntry {
  id: string;
  timestamp_utc: string;
  time_display: string;
  event_type: 'first_seen' | 'observed' | 'moved' | 'disappeared' | 'reappeared' | 'last_seen' | string;
  class_name: string;
  camera_id: string;
  zone_id: string;
  track_id: string;
  confidence: number;
  snapshot_uri: string;
  crop_uri?: string | null;
  gap_duration_sec?: number | null;
  note?: string | null;
}

export interface CameraInfo {
  camera_id: string;
  name: string;
  zone_id: string;
  status: string;
  fps: number;
  resolution: string;
  stream_url: string;
  last_frame_time?: string | null;
}

export interface WorkerStatus {
  worker_id: string;
  name: string;
  status: string;
  fps: number;
  latency_ms: number;
  queue_lag: number;
  processed_count: number;
  last_heartbeat: string;
}

export interface SystemObservability {
  cpu_usage_pct: number;
  memory_usage_pct: number;
  gpu_usage_pct?: number | null;
  p50_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  total_events: number;
  unique_objects_count: number;
  active_tracks: number;
  active_cameras: number;
  dropped_frames: number;
  queue_lag: number;
  workers: WorkerStatus[];
}

export interface OverviewStats {
  total_sightings: number;
  unique_objects_count: number;
  unique_objects: string[];
  active_cameras_count: number;
  active_tracks_count: number;
  processing_fps: number;
  queue_lag: number;
  dropped_frames: number;
  zones: string[];
  vector_count: number;
}

export interface ModelItem {
  id: string;
  name: string;
  file: string;
  description: string;
  is_active: boolean;
  exists: boolean;
  modified_at?: string | null;
  mAP50?: number | null;
}

export interface TrainingState {
  status: 'idle' | 'training' | 'completed' | 'error';
  progress_pct: number;
  current_epoch: number;
  total_epochs: number;
  mAP50: number;
  mAP50_95?: number;
  precision?: number;
  recall?: number;
  box_loss?: number;
  cls_loss?: number;
  message: string;
  logs: string[];
  updated_at: string;
}

export interface ModelsResponse {
  active_model: string;
  active_model_path: string;
  is_custom_trained: boolean;
  models: ModelItem[];
  training_state: TrainingState;
}

export interface CameraStatus {
  source: string;
  camera_name?: string;
  is_synthetic: boolean;
  sample_fps: number;
  queue_size: number;
  healthy: boolean;
  total_frames: number;
  dropped_frames: number;
}
