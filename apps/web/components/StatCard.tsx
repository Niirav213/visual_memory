import React from 'react';
import { LucideIcon } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  badge?: string;
  badgeColor?: 'cyan' | 'emerald' | 'amber' | 'indigo' | 'purple';
}

export default function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  badge,
  badgeColor = 'cyan',
}: StatCardProps) {
  const colorMap = {
    cyan: 'from-cyan-500/20 to-cyan-500/0 text-cyan-400 border-cyan-500/30',
    emerald: 'from-emerald-500/20 to-emerald-500/0 text-emerald-400 border-emerald-500/30',
    amber: 'from-amber-500/20 to-amber-500/0 text-amber-400 border-amber-500/30',
    indigo: 'from-indigo-500/20 to-indigo-500/0 text-indigo-400 border-indigo-500/30',
    purple: 'from-purple-500/20 to-purple-500/0 text-purple-400 border-purple-500/30',
  };

  return (
    <div className="dashboard-card relative overflow-hidden rounded-2xl p-5 group">
      {/* Subtle top glow */}
      <div className={`absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r ${colorMap[badgeColor]}`} />

      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
          {title}
        </span>
        <div className={`p-2.5 rounded-xl bg-slate-800/70 ${colorMap[badgeColor].split(' ')[2]}`}>
          <Icon className="w-4 h-4" aria-hidden="true" />
        </div>
      </div>

      <div className="flex items-baseline space-x-2">
        <span className="text-2xl sm:text-3xl font-extrabold text-white font-mono tracking-tight">
          {value}
        </span>
        {badge && (
          <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full border bg-slate-800 ${colorMap[badgeColor]}`}>
            {badge}
          </span>
        )}
      </div>

      {subtitle && (
        <p className="mt-2 text-xs text-slate-400">
          {subtitle}
        </p>
      )}
    </div>
  );
}
