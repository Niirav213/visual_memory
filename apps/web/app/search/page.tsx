'use client';

import React, { useState, useEffect } from 'react';
import {
  Search,
  Filter,
  MapPin,
  Calendar,
  Clock,
  Sparkles,
  Camera,
  Layers,
  CheckCircle2,
} from 'lucide-react';
import { api } from '../../lib/api';
import { MemoryEvent } from '../../lib/types';
import EvidenceModal from '../../components/EvidenceModal';

export default function SearchPage() {
  const [queryText, setQueryText] = useState<string>('');
  const [selectedClass, setSelectedClass] = useState<string>('');
  const [selectedZone, setSelectedZone] = useState<string>('');
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [events, setEvents] = useState<MemoryEvent[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedEvent, setSelectedEvent] = useState<MemoryEvent | null>(null);

  const objectCategories = [
    'laptop',
    'cell phone',
    'bottle',
    'cup',
    'backpack',
    'book',
    'remote',
    'mouse',
    'keyboard',
  ];

  const zones = [
    'study_desk',
    'kitchen_counter',
    'living_room_table',
    'entryway_shelf',
  ];

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setLoading(true);
    try {
      const results = await api.searchEvents({
        query_text: queryText || undefined,
        object_class: selectedClass || undefined,
        zone_id: selectedZone || undefined,
        camera_id: selectedCamera || undefined,
        limit: 40,
      });
      setEvents(results);
    } catch (err) {
      console.error('Search failed:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    handleSearch();
  }, [selectedClass, selectedZone, selectedCamera]);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5">
        <div className="flex items-center space-x-2">
          <span className="p-1.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
            <Search className="w-4 h-4" />
          </span>
          <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
            Visual Memory Search
          </h1>
        </div>
        <p className="mt-1 text-xs sm:text-sm text-slate-400">
          Query object memories across spatial zones, timestamps, and semantic OpenCLIP visual embeddings.
        </p>
      </div>

      {/* Search Input Bar */}
      <form onSubmit={handleSearch} className="relative">
        <div className="relative flex items-center">
          <Search className="absolute left-4 w-5 h-5 text-slate-400 pointer-events-none" />
          <input
            type="text"
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            placeholder="Search by object description (e.g. 'laptop on study desk', 'red bottle near window')..."
            className="w-full pl-12 pr-28 py-3.5 rounded-xl bg-slate-900 border border-slate-800 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all shadow-lg"
          />
          <button
            type="submit"
            className="absolute right-2 px-4 py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-indigo-600 text-white text-xs font-semibold hover:from-cyan-400 hover:to-indigo-500 transition-all shadow-md shadow-cyan-500/20"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </form>

      {/* Filter Row: Category Chips & Zone Selectors */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-slate-400 mr-2 flex items-center space-x-1">
          <Filter className="w-3.5 h-3.5" />
          <span>Object:</span>
        </span>
        <button
          onClick={() => setSelectedClass('')}
          className={`px-3 py-1 rounded-lg text-xs font-mono transition-all ${
            selectedClass === ''
              ? 'bg-cyan-500/20 border border-cyan-500/40 text-cyan-300 font-semibold'
              : 'bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200'
          }`}
        >
          ALL OBJECTS
        </button>
        {objectCategories.map((cls) => (
          <button
            key={cls}
            onClick={() => setSelectedClass(selectedClass === cls ? '' : cls)}
            className={`px-3 py-1 rounded-lg text-xs font-mono transition-all ${
              selectedClass === cls
                ? 'bg-cyan-500/20 border border-cyan-500/40 text-cyan-300 font-semibold'
                : 'bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            {cls.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Secondary Dropdown Filters */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 p-4 rounded-xl bg-slate-900/40 border border-slate-800 text-xs">
        <div>
          <label className="block text-slate-400 mb-1 font-mono">Zone Filter</label>
          <select
            value={selectedZone}
            onChange={(e) => setSelectedZone(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500"
          >
            <option value="">Any Zone</option>
            {zones.map((z) => (
              <option key={z} value={z}>{z}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-slate-400 mb-1 font-mono">Camera Node</label>
          <select
            value={selectedCamera}
            onChange={(e) => setSelectedCamera(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500"
          >
            <option value="">Any Camera</option>
            <option value="camera_01">camera_01 (Desk Main)</option>
            <option value="camera_02">camera_02 (Kitchen/Entryway)</option>
          </select>
        </div>

        <div className="flex items-end">
          <button
            onClick={() => {
              setSelectedClass('');
              setSelectedZone('');
              setSelectedCamera('');
              setQueryText('');
            }}
            className="w-full py-2 px-3 rounded-lg border border-slate-800 bg-slate-800/60 text-slate-300 hover:bg-slate-800 transition-colors font-mono"
          >
            Reset Filters
          </button>
        </div>
      </div>

      {/* Results Grid */}
      <div className="space-y-3">
        <div className="flex items-center justify-between text-xs font-mono text-slate-400">
          <span>Found {events.length} Evidence Sightings</span>
          <span>Sorted by Timestamp (Latest First)</span>
        </div>

        {events.length === 0 ? (
          <div className="text-center py-16 rounded-xl border border-slate-800 bg-slate-900/30">
            <Search className="w-8 h-8 text-slate-600 mx-auto mb-2" />
            <p className="text-sm font-medium text-slate-300">No matching visual memories found</p>
            <p className="text-xs text-slate-500 mt-1">Try adjusting your filters or search keywords</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {events.map((evt) => (
              <div
                key={evt.event_id}
                onClick={() => setSelectedEvent(evt)}
                className="group cursor-pointer rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden hover:border-cyan-500/40 hover:bg-slate-900/90 transition-all shadow-md"
              >
                {/* Snapshot Image with Zone Badge */}
                <div className="relative aspect-video w-full bg-slate-950 overflow-hidden">
                  <img
                    src={evt.snapshot_uri}
                    alt={evt.class_name}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    onError={(e) => {
                      (e.target as HTMLElement).style.display = 'none';
                    }}
                  />
                  <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/80 text-[10px] font-mono text-cyan-300 border border-slate-800">
                    {evt.zone_id}
                  </div>
                  <div className="absolute top-2 right-2 px-1.5 py-0.5 rounded bg-black/80 text-[10px] font-mono text-emerald-400 border border-slate-800">
                    {(evt.confidence * 100).toFixed(0)}%
                  </div>
                </div>

                {/* Card Details */}
                <div className="p-3.5 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-white uppercase text-sm">{evt.class_name}</span>
                    <span className="text-[11px] font-mono text-slate-400">#{evt.track_id}</span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
                    <span>{evt.camera_id}</span>
                    <span>{evt.timestamp_utc.slice(11, 19)} UTC</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <EvidenceModal
        event={selectedEvent}
        onClose={() => setSelectedEvent(null)}
      />
    </div>
  );
}
