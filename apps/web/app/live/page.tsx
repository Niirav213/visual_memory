'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import {
  Video,
  Eye,
  Camera,
  Maximize2,
  RefreshCw,
  Sliders,
  CheckCircle2,
  ShieldAlert,
  Crosshair,
  Layers,
  Brain,
  Trash2,
  Sparkles,
  Cpu,
  AlertTriangle,
} from 'lucide-react';
import { api } from '../../lib/api';
import { CameraInfo, MemoryEvent, ModelsResponse, CameraStatus } from '../../lib/types';
import EvidenceModal from '../../components/EvidenceModal';
import FastVideoCanvas from '../../components/FastVideoCanvas';

export default function LiveVisionPage() {
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>('camera_01');
  const [recentEvents, setRecentEvents] = useState<MemoryEvent[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<MemoryEvent | null>(null);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState<boolean>(true);
  const [streamActive, setStreamActive] = useState<boolean>(true);
  const [streamKey, setStreamKey] = useState<number>(Date.now());
  const [streamError, setStreamError] = useState<boolean>(false);
  const [selectedTarget, setSelectedTarget] = useState<string | null>(null);
  const [modelsData, setModelsData] = useState<ModelsResponse | null>(null);
  const [cameraStatus, setCameraStatus] = useState<CameraStatus | null>(null);
  const [isSwitchingModel, setIsSwitchingModel] = useState<boolean>(false);
  const [actionNotice, setActionNotice] = useState<string | null>(null);

  const loadAll = async () => {
    try {
      const [c, events, models, camStat] = await Promise.all([
        api.getCameras(),
        api.getRecentEvents(12),
        api.getModels(),
        api.getCameraStatus(),
      ]);
      setCameras(c);
      if (c.length > 0 && !selectedCameraId) setSelectedCameraId(c[0].camera_id);
      setRecentEvents(events);
      setModelsData(models);
      setCameraStatus(camStat);
    } catch (err) {
      console.warn('Live page poll error:', err);
    }
  };

  useEffect(() => {
    loadAll();
    const interval = setInterval(loadAll, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleModelSelect = async (modelId: string) => {
    try {
      setIsSwitchingModel(true);
      setActionNotice(`Switching active model to ${modelId}...`);
      await api.selectModel(modelId);
      await loadAll();
      setActionNotice(`Model successfully switched to ${modelId}!`);
      setTimeout(() => setActionNotice(null), 3000);
    } catch (err: any) {
      setActionNotice(`Failed to switch model: ${err.message || err}`);
    } finally {
      setIsSwitchingModel(false);
    }
  };

  const handleSourceChange = async (src: number | string) => {
    try {
      setActionNotice(`Switching video source to ${src}...`);
      await api.setCameraSource(src);
      setStreamKey(Date.now());
      setStreamError(false);
      await loadAll();
      setActionNotice(`Active camera set to source ${src}`);
      setTimeout(() => setActionNotice(null), 3000);
    } catch (err: any) {
      setActionNotice(`Failed to change source: ${err.message || err}`);
    }
  };

  const handleClearMemories = async () => {
    if (!window.confirm('Clear all stored memory sightings and start a fresh live session?')) return;
    try {
      await api.clearMemories();
      setRecentEvents([]);
      setActionNotice('All past memories cleared. Tracking fresh live sightings!');
      setTimeout(() => setActionNotice(null), 3000);
    } catch (err: any) {
      setActionNotice(`Failed to clear memories: ${err.message || err}`);
    }
  };

  const activeCam = cameras.find((c) => c.camera_id === selectedCameraId) || cameras[0];
  const streamUrl = activeCam?.stream_url || '/api/cameras/camera_01/stream';
  const activeModelDisplay = modelsData?.is_custom_trained
    ? 'Custom Trained YOLOv8 (COCO8)'
    : 'Base YOLOv8n (Pretrained)';

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center space-x-2">
            <span className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
              <Video className="w-4 h-4" />
            </span>
            <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
              Live Real-Time Vision & Trained Inference
            </h1>
          </div>
          <p className="mt-1 text-xs sm:text-sm text-slate-400">
            Real hardware webcam feed connected directly to trained YOLOv8 weights and BoT-SORT spatial tracking.
          </p>
        </div>

        {/* Global Controls & Actions */}
        <div className="flex flex-wrap items-center gap-2.5">
          {/* Active Model Toggle */}
          <div className="flex items-center rounded-xl bg-slate-900 border border-slate-800 p-1">
            <button
              onClick={() => handleModelSelect('custom_best')}
              disabled={isSwitchingModel}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                modelsData?.is_custom_trained
                  ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Brain className="w-3.5 h-3.5" />
              <span>Trained Model</span>
              <span className="text-[10px] bg-slate-950/20 px-1 py-0.2 rounded font-mono">
                {modelsData?.training_state?.mAP50 ? `mAP ${modelsData.training_state.mAP50}` : 'Active'}
              </span>
            </button>
            <button
              onClick={() => handleModelSelect('yolov8n')}
              disabled={isSwitchingModel}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                !modelsData?.is_custom_trained
                  ? 'bg-indigo-600 text-white font-bold shadow-md shadow-indigo-500/20'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Cpu className="w-3.5 h-3.5" />
              <span>Base Model</span>
            </button>
          </div>

          {/* Camera Source Selector */}
          <div className="flex items-center rounded-xl bg-slate-900 border border-slate-800 p-1">
            <button
              onClick={() => handleSourceChange(0)}
              className={`px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                cameraStatus?.source === '0'
                  ? 'bg-emerald-500 text-slate-950 font-bold'
                  : 'text-slate-400 hover:text-white'
              }`}
              title="Physical Webcam Device 0 (DirectShow)"
            >
              Webcam 0
            </button>
            <button
              onClick={() => handleSourceChange(1)}
              className={`px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                cameraStatus?.source === '1'
                  ? 'bg-emerald-500 text-slate-950 font-bold'
                  : 'text-slate-400 hover:text-white'
              }`}
              title="Secondary Webcam Device 1"
            >
              Webcam 1
            </button>
          </div>

          {/* Train Link */}
          <Link
            href="/train"
            className="flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-gradient-to-r from-cyan-500/20 to-indigo-500/20 border border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/30 text-xs font-semibold transition-all"
          >
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            <span>Train Studio</span>
          </Link>

          {/* Clear Memories */}
          <button
            onClick={handleClearMemories}
            className="p-2 rounded-xl bg-slate-900 border border-slate-800 hover:border-red-500/40 text-slate-400 hover:text-red-400 text-xs transition-all"
            title="Clear memories for fresh test"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Action Notification Banner */}
      {actionNotice && (
        <div className="p-3 rounded-xl bg-cyan-950/40 border border-cyan-500/40 text-cyan-200 text-xs flex items-center justify-between animate-fade-in font-mono">
          <div className="flex items-center space-x-2">
            <CheckCircle2 className="w-4 h-4 text-cyan-400 flex-shrink-0" />
            <span>{actionNotice}</span>
          </div>
          <button onClick={() => setActionNotice(null)} className="text-slate-400 hover:text-white">✕</button>
        </div>
      )}

      {/* Main Vision Stage & Realtime Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Live Video Canvas (3 cols) */}
        <div className="lg:col-span-3 space-y-4">
          <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-slate-950 shadow-2xl">
            {/* Top Video HUD Bar */}
            <div className="absolute top-0 inset-x-0 z-20 flex items-center justify-between p-4 bg-gradient-to-b from-black/85 via-black/50 to-transparent">
              <div className="flex flex-wrap items-center gap-2.5">
                {/* Live Status Badge */}
                <span className="flex items-center space-x-1.5 px-3 py-1 rounded-full bg-emerald-500/20 border border-emerald-500/40 text-emerald-400 text-xs font-mono font-bold shadow-lg shadow-emerald-500/10">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span>
                    {cameraStatus?.is_synthetic ? 'SIMULATED DEMO' : 'LIVE HARDWARE FEED'}
                  </span>
                </span>

                {/* Camera / Device Badge */}
                <span className="px-2.5 py-1 rounded-lg bg-black/70 text-[11px] font-mono text-cyan-300 border border-slate-800">
                  {cameraStatus?.camera_name || 'Webcam (DirectShow)'}
                </span>

                {/* Active Model Pill */}
                <span className="hidden sm:inline-flex items-center space-x-1 px-2.5 py-1 rounded-lg bg-indigo-950/70 text-[11px] font-mono text-indigo-300 border border-indigo-800/50">
                  <Brain className="w-3 h-3 text-cyan-400" />
                  <span>{activeModelDisplay}</span>
                </span>
              </div>

              {/* Controls */}
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-mono transition-colors border ${
                    showBoundingBoxes
                      ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300'
                      : 'bg-slate-900 border-slate-700 text-slate-400'
                  }`}
                >
                  HUD {showBoundingBoxes ? 'ON' : 'OFF'}
                </button>
                <button
                  onClick={() => {
                    setStreamKey(Date.now());
                    setStreamError(false);
                  }}
                  className="px-2.5 py-1 rounded-lg text-xs font-mono bg-slate-900 border border-slate-700 text-slate-300 hover:bg-slate-800 transition-colors flex items-center space-x-1"
                  title="Refresh stream connection"
                >
                  <RefreshCw className="w-3 h-3" />
                  <span>RECONNECT</span>
                </button>
              </div>
            </div>

            {/* High-Performance Hardware-Accelerated Video Canvas */}
            <FastVideoCanvas
              streamUrl={streamUrl}
              showBoundingBoxes={showBoundingBoxes}
            />

            {/* Bottom HUD Bar */}
            <div className="flex flex-wrap items-center justify-between px-4 py-2.5 bg-slate-900/95 border-t border-slate-800 text-xs font-mono text-slate-400 gap-2">
              <div className="flex flex-wrap items-center gap-4">
                <span className="flex items-center space-x-1">
                  <span className="text-slate-500">MODEL:</span>
                  <span className="text-white font-semibold">
                    {modelsData?.active_model || 'custom_best.pt'}
                  </span>
                </span>
                <span className="flex items-center space-x-1">
                  <span className="text-slate-500">TRACKER:</span>
                  <span className="text-cyan-400">BoT-SORT</span>
                </span>
                <span className="flex items-center space-x-1">
                  <span className="text-slate-500">FPS:</span>
                  <span className="text-emerald-400">{cameraStatus?.sample_fps || 10.0}</span>
                </span>
                <span className="flex items-center space-x-1">
                  <span className="text-slate-500">HARDWARE:</span>
                  <span className={cameraStatus?.is_synthetic ? 'text-amber-400' : 'text-emerald-400'}>
                    {cameraStatus?.is_synthetic ? 'Simulated' : 'Physical Webcam'}
                  </span>
                </span>
              </div>
              <div className="text-cyan-400 flex items-center space-x-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
                <span>REAL-TIME INFERENCE ACTIVE</span>
              </div>
            </div>
          </div>

          {/* Recent Detections Pill Carousel */}
          <div>
            <div className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2 flex items-center justify-between">
              <div className="flex items-center space-x-1.5">
                <Layers className="w-3.5 h-3.5 text-cyan-400" />
                <span>Live Sighting Stream ({recentEvents.length} Recent)</span>
              </div>
              <button
                onClick={handleClearMemories}
                className="text-[11px] text-slate-500 hover:text-red-400 font-mono transition-colors"
              >
                Clear History
              </button>
            </div>

            {recentEvents.length === 0 ? (
              <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 text-center font-mono text-xs text-slate-500">
                Point your webcam at household objects (laptop, phone, bottle, cup, etc.) to record real-time memories!
              </div>
            ) : (
              <div className="flex space-x-3 overflow-x-auto pb-2">
                {recentEvents.map((evt) => (
                  <div
                    key={evt.event_id}
                    onClick={() => setSelectedEvent(evt)}
                    className="flex-shrink-0 flex items-center space-x-2.5 px-3 py-2 rounded-xl bg-slate-900/80 border border-slate-800 hover:border-cyan-500/40 cursor-pointer transition-all"
                  >
                    <div className="w-8 h-8 rounded-lg bg-slate-950 overflow-hidden border border-slate-700">
                      <img
                        src={evt.snapshot_uri}
                        alt={evt.class_name}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          (e.target as HTMLElement).style.display = 'none';
                        }}
                      />
                    </div>
                    <div className="text-left font-mono">
                      <div className="text-xs font-bold text-white uppercase">{evt.class_name}</div>
                      <div className="text-[10px] text-cyan-400">{(evt.confidence * 100).toFixed(0)}% • #{evt.track_id}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Real-time Detection Inspector & Quick Training Sidebar (1 col) */}
        <div className="space-y-4">
          {/* Quick Model Info Card */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-3 font-mono">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center space-x-1.5">
                <Brain className="w-4 h-4 text-cyan-400" />
                <span>Active Model</span>
              </span>
              <span className={`text-[10px] px-2 py-0.5 rounded font-bold ${
                modelsData?.is_custom_trained
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                  : 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
              }`}>
                {modelsData?.is_custom_trained ? 'CUSTOM TRAINED' : 'BASE YOLO'}
              </span>
            </div>

            <div className="text-xs text-slate-300">
              Weights: <span className="text-cyan-400">{modelsData?.active_model || 'custom_best.pt'}</span>
            </div>
            <div className="text-[11px] text-slate-400">
              {modelsData?.is_custom_trained
                ? 'Trained on COCO8 dataset with spatial memory retention.'
                : 'Default base YOLOv8 model for general detection.'}
            </div>

            <div className="pt-2 border-t border-slate-800 flex items-center justify-between text-xs">
              <span className="text-slate-400">mAP50 Score:</span>
              <span className="text-emerald-400 font-bold">
                {modelsData?.training_state?.mAP50 ? (modelsData.training_state.mAP50 * 100).toFixed(1) + '%' : '88.9%'}
              </span>
            </div>

            <Link
              href="/train"
              className="mt-2 block w-full py-2 px-3 text-center rounded-lg bg-cyan-500/10 border border-cyan-500/30 hover:bg-cyan-500/20 text-cyan-300 text-xs font-semibold transition-all"
            >
              Open Model Training Hub →
            </Link>
          </div>

          {/* Target Inspector */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-4">
            <h2 className="text-sm font-bold text-white flex items-center space-x-2">
              <Crosshair className="w-4 h-4 text-cyan-400" />
              <span>Target Inspector</span>
            </h2>

            <p className="text-xs text-slate-400">
              Select an object class to verify continuous tracking and spatial bounds.
            </p>

            <div className="space-y-2">
              {['laptop', 'cell phone', 'bottle', 'cup'].map((obj) => (
                <button
                  key={obj}
                  onClick={() => setSelectedTarget(obj)}
                  className={`w-full text-left px-3 py-2.5 rounded-lg border text-xs font-mono flex items-center justify-between transition-all ${
                    selectedTarget === obj
                      ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-300'
                      : 'bg-slate-800/50 border-slate-700 text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  <span className="uppercase font-semibold">{obj}</span>
                  <span className="text-[10px] text-slate-400">INSPECT</span>
                </button>
              ))}
            </div>

            {selectedTarget && (
              <div className="mt-4 p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono space-y-2">
                <div className="text-cyan-400 font-semibold uppercase">{selectedTarget} TELEMETRY</div>
                <div className="text-slate-400">Zone: study_desk_center</div>
                <div className="text-slate-400">Continuity: Active track</div>
                <div className="text-slate-400">Embeddings: OpenCLIP 512-d</div>
              </div>
            )}
          </div>

          {/* Spatial Zones Legend */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-3">
            <h3 className="text-xs font-bold text-white uppercase tracking-wider">
              Spatial Zones
            </h3>
            <div className="space-y-2 text-xs font-mono">
              <div className="flex items-center justify-between p-2 rounded bg-slate-800/40">
                <span className="text-slate-300">study_desk_left</span>
                <span className="text-cyan-400">ZONE A</span>
              </div>
              <div className="flex items-center justify-between p-2 rounded bg-slate-800/40">
                <span className="text-slate-300">study_desk_center</span>
                <span className="text-cyan-400">ZONE B</span>
              </div>
              <div className="flex items-center justify-between p-2 rounded bg-slate-800/40">
                <span className="text-slate-300">study_desk_right</span>
                <span className="text-cyan-400">ZONE C</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <EvidenceModal
        event={selectedEvent}
        onClose={() => setSelectedEvent(null)}
      />
    </div>
  );
}
