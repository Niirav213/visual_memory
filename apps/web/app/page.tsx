'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  Eye,
  Camera,
  Layers,
  Activity,
  Cpu,
  Database,
  ArrowRight,
  Sparkles,
  Search,
  Clock,
  CheckCircle2,
  AlertCircle,
  ExternalLink,
  ImageOff,
} from 'lucide-react';
import StatCard from '../components/StatCard';
import EvidenceModal from '../components/EvidenceModal';
import { api } from '../lib/api';
import { OverviewStats, MemoryEvent, CameraInfo } from '../lib/types';

export default function OverviewPage() {
  const [stats, setStats] = useState<OverviewStats | null>(null);
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [recentEvents, setRecentEvents] = useState<MemoryEvent[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<MemoryEvent | null>(null);
  const [loading, setLoading] = useState(true);

  const loadData = async () => {
    try {
      const [s, c, e] = await Promise.all([
        api.getOverview(),
        api.getCameras(),
        api.getRecentEvents(8),
      ]);
      setStats(s);
      setCameras(c);
      setRecentEvents(e);
    } catch (err) {
      console.error('Error loading overview data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-7 sm:space-y-9 animate-fade-in">
      {/* Top Banner / Hero */}
      <div className="hero-panel p-6 sm:p-9 lg:p-11">
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-800/70 border border-slate-600/50 text-amber-100 text-[10px] sm:text-[11px] font-medium tracking-[0.12em] mb-5">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-200" />
            <span>OBJECT-CENTRIC VISUAL MEMORY</span>
          </div>
          <h1 className="text-3xl sm:text-4xl lg:text-[2.8rem] font-semibold text-white tracking-tight leading-[1.12]">
            A clearer view of <span className="text-amber-100">what you saw.</span>
          </h1>
          <p className="mt-3 max-w-2xl text-sm sm:text-base leading-7 text-slate-400">
            Find everyday objects across your visual history, with each answer grounded in a time, place, and image.
          </p>

          <div className="mt-7 flex flex-wrap gap-3">
            <Link
              href="/ask"
              className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-amber-100 text-slate-950 font-semibold text-xs hover:bg-amber-200 transition-colors shadow-lg shadow-amber-900/20"
            >
              <Sparkles className="w-4 h-4" />
              <span>Ask Visual Memory</span>
            </Link>
            <Link
              href="/live"
              className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-white/[0.04] border border-white/10 text-slate-200 font-medium text-xs hover:bg-white/[0.08] transition-colors"
            >
              <Eye className="w-4 h-4 text-cyan-400" />
              <span>Live Vision Feed</span>
            </Link>
            <Link
              href="/search"
              className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-white/[0.04] border border-white/10 text-slate-200 font-medium text-xs hover:bg-white/[0.08] transition-colors"
            >
              <Search className="w-4 h-4 text-indigo-400" />
              <span>Memory Search</span>
            </Link>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Sightings"
          value={stats?.total_sightings ?? 0}
          subtitle="Persistent visual memories"
          icon={Layers}
          badge="+Live"
          badgeColor="cyan"
        />
        <StatCard
          title="Unique Objects"
          value={stats?.unique_objects_count ?? 0}
          subtitle="Distinct everyday categories"
          icon={Database}
          badge="Tracked"
          badgeColor="emerald"
        />
        <StatCard
          title="Processing FPS"
          value={stats?.processing_fps ?? 4.8}
          subtitle="Decoupled inference loop"
          icon={Activity}
          badge="Healthy"
          badgeColor="indigo"
        />
        <StatCard
          title="Active Cameras"
          value={stats?.active_cameras_count ?? 1}
          subtitle="Monitored spatial zones"
          icon={Camera}
          badge="Streaming"
          badgeColor="amber"
        />
      </div>

      {/* Main Grid: Active Cameras & Recent Sighting Stream */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
        {/* Left Column: Active Camera Streams */}
        <section className="dashboard-section-panel space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-bold text-white flex items-center space-x-2">
              <Camera className="w-4 h-4 text-cyan-400" />
              <span>Active Camera Nodes</span>
            </h2>
            <Link href="/live" className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center space-x-1">
              <span>View Full</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-3.5">
            {cameras.map((cam) => (
              <div
                key={cam.camera_id}
                className="dashboard-card camera-card flex h-[clamp(260px,22vw,340px)] min-w-0 flex-col overflow-hidden rounded-xl p-4"
              >
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h3 className="font-semibold text-white text-sm">{cam.name}</h3>
                    <p className="text-xs text-slate-400 font-mono">Zone: {cam.zone_id}</p>
                  </div>
                  <span className="flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-mono bg-emerald-500/15 border border-emerald-500/30 text-emerald-400">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    <span>{cam.status.toUpperCase()}</span>
                  </span>
                </div>

                <div className="relative mt-auto min-h-0 w-full flex-1 rounded-lg bg-black/35 border border-white/10 overflow-hidden flex items-center justify-center">
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-center text-xs text-slate-400">
                    <ImageOff className="h-5 w-5 text-slate-500" aria-hidden="true" />
                    <span>Live preview unavailable</span>
                  </div>
                  <img
                    src={cam.stream_url}
                    alt={cam.name}
                    className="absolute inset-0 h-full w-full object-cover"
                    onError={(e) => {
                      (e.target as HTMLElement).style.display = 'none';
                    }}
                  />
                  <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/70 text-[10px] font-mono text-cyan-400 border border-slate-800">
                    {cam.fps} FPS // {cam.resolution}
                  </div>
                </div>
              </div>
            ))}
          </div>

        </section>

        {/* Right 2 Columns: Recent Memory Sightings Stream */}
        <section className="dashboard-section-panel space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-bold text-white flex items-center space-x-2">
              <Clock className="w-4 h-4 text-indigo-400" />
              <span>Recent Sighting Events</span>
            </h2>
            <Link href="/timeline" className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center space-x-1">
              <span>Full Timeline</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-3.5">
            {recentEvents.map((evt) => (
              <div
                key={evt.event_id}
                onClick={() => setSelectedEvent(evt)}
                className="dashboard-card sighting-card group flex h-[clamp(260px,22vw,340px)] min-w-0 cursor-pointer flex-col rounded-xl p-4"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-cyan-400 group-hover:scale-125 transition-transform" />
                    <span className="font-bold text-slate-100 uppercase tracking-wide text-sm">
                      {evt.class_name}
                    </span>
                  </div>
                  <span className="text-[11px] font-mono text-cyan-300 px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/20">
                    {(evt.confidence * 100).toFixed(0)}%
                  </span>
                </div>

                <div className="relative mt-auto min-h-0 w-full flex-1 rounded-lg bg-black/35 border border-white/10 overflow-hidden mb-3">
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-center text-xs text-slate-400">
                    <ImageOff className="h-5 w-5 text-slate-500" aria-hidden="true" />
                    <span>Snapshot unavailable</span>
                  </div>
                  <img
                    src={evt.snapshot_uri}
                    alt={evt.class_name}
                    className="absolute inset-0 h-full w-full object-cover group-hover:scale-105 transition-transform duration-300"
                    onError={(e) => {
                      (e.target as HTMLElement).style.display = 'none';
                    }}
                  />
                  <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/75 text-[10px] font-mono text-slate-300 border border-slate-800">
                    {evt.zone_id}
                  </div>
                </div>

                <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
                  <span>Track #{evt.track_id}</span>
                  <span>{evt.timestamp_utc.slice(11, 19)} UTC</span>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>

      <aside className="scope-policy-panel p-4 sm:px-5 sm:py-4 text-xs text-slate-200 space-y-1.5">
        <div className="font-semibold text-white flex items-center space-x-1.5">
          <CheckCircle2 className="w-3.5 h-3.5 text-amber-200" />
          <span>Object-Centric Scope Policy</span>
        </div>
        <p className="text-slate-200/85">
          Strictly focused on physical everyday items (laptops, phones, cups, keys). Pedestrian tracking and person re-ID features are excluded by architecture design.
        </p>
      </aside>

      {/* Evidence Inspection Modal */}
      <EvidenceModal
        event={selectedEvent}
        onClose={() => setSelectedEvent(null)}
      />
    </div>
  );
}
