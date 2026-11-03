'use client';

import React, { useState, useEffect } from 'react';
import {
  Clock,
  ArrowRight,
  Eye,
  Move,
  AlertTriangle,
  RotateCcw,
  Sparkles,
  MapPin,
  Camera,
  Filter,
} from 'lucide-react';
import { api } from '../../lib/api';
import { TimelineEntry, MemoryEvent } from '../../lib/types';
import EvidenceModal from '../../components/EvidenceModal';

export default function TimelinePage() {
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);
  const [selectedClass, setSelectedClass] = useState<string>('');
  const [selectedEvent, setSelectedEvent] = useState<MemoryEvent | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const objectOptions = ['laptop', 'cell phone', 'bottle', 'cup', 'backpack', 'book', 'remote'];

  const loadTimeline = async () => {
    setLoading(true);
    try {
      const data = await api.getTimeline(selectedClass || undefined);
      setTimeline(data);
    } catch (err) {
      console.error('Failed to load timeline:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTimeline();
  }, [selectedClass]);

  const getEventBadge = (type: string) => {
    switch (type) {
      case 'moved':
        return {
          icon: Move,
          color: 'bg-amber-500/15 border-amber-500/30 text-amber-300',
          dot: 'bg-amber-400',
          label: 'OBJECT MOVED',
        };
      case 'disappeared':
        return {
          icon: AlertTriangle,
          color: 'bg-rose-500/15 border-rose-500/30 text-rose-300',
          dot: 'bg-rose-400',
          label: 'VISIBILITY GAP',
        };
      case 'reappeared':
        return {
          icon: RotateCcw,
          color: 'bg-cyan-500/15 border-cyan-500/30 text-cyan-300',
          dot: 'bg-cyan-400',
          label: 'REAPPEARED',
        };
      case 'first_seen':
        return {
          icon: Sparkles,
          color: 'bg-purple-500/15 border-purple-500/30 text-purple-300',
          dot: 'bg-purple-400',
          label: 'FIRST RECORDED',
        };
      default:
        return {
          icon: Eye,
          color: 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300',
          dot: 'bg-emerald-400',
          label: 'OBSERVED',
        };
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center space-x-2">
            <span className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
              <Clock className="w-4 h-4" />
            </span>
            <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
              Memory Chronological Timeline
            </h1>
          </div>
          <p className="mt-1 text-xs sm:text-sm text-slate-400">
            Chronological log of object appearances, inferred visibility gaps, zone transitions, and evidence snapshots.
          </p>
        </div>

        {/* Filter */}
        <div className="flex items-center space-x-2">
          <Filter className="w-4 h-4 text-slate-400" />
          <select
            value={selectedClass}
            onChange={(e) => setSelectedClass(e.target.value)}
            className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500"
          >
            <option value="">All Object Classes</option>
            {objectOptions.map((o) => (
              <option key={o} value={o}>{o.toUpperCase()}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Timeline Stream */}
      <div className="relative pl-6 sm:pl-8 space-y-6 before:absolute before:left-3 before:top-2 before:bottom-2 before:w-0.5 before:bg-slate-800">
        {timeline.map((item, idx) => {
          const badge = getEventBadge(item.event_type);
          const Icon = badge.icon;

          return (
            <div key={item.id} className="relative group">
              {/* Timeline Indicator Dot */}
              <div
                className={`absolute -left-6 sm:-left-8 top-3.5 w-4 h-4 rounded-full border-2 border-slate-950 ${badge.dot} shadow-md shadow-cyan-500/20 group-hover:scale-125 transition-transform`}
              />

              {/* Event Card */}
              <div
                onClick={() =>
                  setSelectedEvent({
                    event_id: item.id,
                    camera_id: item.camera_id,
                    zone_id: item.zone_id,
                    track_id: item.track_id,
                    class_name: item.class_name,
                    confidence: item.confidence,
                    timestamp_utc: item.timestamp_utc,
                    bbox_xyxy: [100, 100, 300, 300],
                    snapshot_uri: item.snapshot_uri,
                    event_label: item.event_type,
                  })
                }
                className="cursor-pointer overflow-hidden rounded-xl border border-slate-800 bg-slate-900/60 p-4 sm:p-5 hover:border-cyan-500/40 hover:bg-slate-900/90 transition-all shadow-md"
              >
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-3">
                  <div className="flex items-center space-x-3">
                    <span className={`inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-mono border ${badge.color}`}>
                      <Icon className="w-3 h-3" />
                      <span>{badge.label}</span>
                    </span>
                    <h3 className="font-bold text-white uppercase text-sm sm:text-base tracking-wide">
                      {item.class_name}
                    </h3>
                  </div>

                  <div className="text-xs font-mono text-slate-400">
                    {item.time_display} • {item.timestamp_utc.slice(11, 19)} UTC
                  </div>
                </div>

                {/* Body: Zone & Note Info */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-3">
                  <div className="sm:col-span-2 space-y-2 text-xs font-mono text-slate-300">
                    <div className="flex items-center space-x-2">
                      <MapPin className="w-3.5 h-3.5 text-cyan-400" />
                      <span>Zone: <strong className="text-white">{item.zone_id}</strong></span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Camera className="w-3.5 h-3.5 text-indigo-400" />
                      <span>Camera: <strong className="text-white">{item.camera_id}</strong> (Track #{item.track_id})</span>
                    </div>
                    {item.note && (
                      <p className="text-xs text-amber-300/90 bg-amber-500/10 border border-amber-500/20 p-2 rounded-lg font-sans">
                        ℹ️ {item.note}
                      </p>
                    )}
                  </div>

                  {/* Thumbnail */}
                  <div className="relative aspect-video w-full rounded-lg bg-slate-950 border border-slate-800 overflow-hidden">
                    <img
                      src={item.snapshot_uri}
                      alt={item.class_name}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                      onError={(e) => {
                        (e.target as HTMLElement).style.display = 'none';
                      }}
                    />
                    <div className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded bg-black/80 text-[9px] font-mono text-cyan-300">
                      VIEW EVIDENCE
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <EvidenceModal
        event={selectedEvent}
        onClose={() => setSelectedEvent(null)}
      />
    </div>
  );
}
