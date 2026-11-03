'use client';

import React from 'react';
import { X, MapPin, Camera, Clock, Crosshair, Sparkles } from 'lucide-react';
import { MemoryEvent } from '../lib/types';

interface EvidenceModalProps {
  event: MemoryEvent | null;
  onClose: () => void;
}

export default function EvidenceModal({ event, onClose }: EvidenceModalProps) {
  if (!event) return null;

  const [x1, y1, x2, y2] = event.bbox_xyxy;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in">
      <div className="relative w-full max-w-3xl overflow-hidden rounded-2xl border border-slate-700 bg-slate-900 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 px-6 py-4">
          <div className="flex items-center space-x-3">
            <span className="flex h-3 w-3 rounded-full bg-cyan-400 animate-pulse" />
            <h3 className="text-base font-semibold text-white">
              Visual Evidence Inspector — <span className="text-cyan-400 uppercase font-mono">{event.class_name}</span>
            </h3>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 space-y-6">
          {/* Frame Display Area */}
          <div className="relative aspect-video w-full overflow-hidden rounded-xl border border-slate-800 bg-slate-950 flex items-center justify-center">
            {event.snapshot_uri ? (
              <img
                src={event.snapshot_uri}
                alt={`Evidence of ${event.class_name}`}
                className="w-full h-full object-contain"
                onError={(e) => {
                  (e.target as HTMLElement).style.display = 'none';
                }}
              />
            ) : null}

            {/* Bounding Box HUD Overlay */}
            <div className="absolute inset-0 pointer-events-none border border-cyan-500/10">
              <div className="absolute top-3 left-3 bg-slate-950/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-cyan-500/30 text-xs font-mono text-cyan-300">
                CAM: {event.camera_id} // ZONE: {event.zone_id.toUpperCase()}
              </div>
              <div className="absolute bottom-3 right-3 bg-slate-950/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-700 text-xs font-mono text-slate-300">
                CONFIDENCE: {(event.confidence * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          {/* Metadata Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
            <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-800">
              <div className="flex items-center space-x-1.5 text-slate-400 mb-1">
                <MapPin className="w-3.5 h-3.5 text-cyan-400" />
                <span>Zone Location</span>
              </div>
              <div className="font-semibold text-slate-100">{event.zone_id}</div>
            </div>

            <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-800">
              <div className="flex items-center space-x-1.5 text-slate-400 mb-1">
                <Camera className="w-3.5 h-3.5 text-indigo-400" />
                <span>Camera Stream</span>
              </div>
              <div className="font-semibold text-slate-100">{event.camera_id}</div>
            </div>

            <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-800">
              <div className="flex items-center space-x-1.5 text-slate-400 mb-1">
                <Crosshair className="w-3.5 h-3.5 text-emerald-400" />
                <span>Track ID</span>
              </div>
              <div className="font-semibold text-slate-100">#{event.track_id}</div>
            </div>

            <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-800">
              <div className="flex items-center space-x-1.5 text-slate-400 mb-1">
                <Clock className="w-3.5 h-3.5 text-amber-400" />
                <span>Timestamp</span>
              </div>
              <div className="font-semibold text-slate-100 truncate">
                {event.timestamp_utc.slice(11, 19)} UTC
              </div>
            </div>
          </div>

          {/* Raw Coordinates & Policy Note */}
          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
            <span className="font-mono">
              BBOX: [{x1}, {y1}, {x2}, {y2}] • EVENT ID: {event.event_id}
            </span>
            <span className="text-cyan-400 font-medium">Evidence Ground Truth</span>
          </div>
        </div>
      </div>
    </div>
  );
}
