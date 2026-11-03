'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Brain,
  Video,
  Search,
  Clock,
  Sparkles,
  Activity,
  Layers,
} from 'lucide-react';

export default function Navbar() {
  const pathname = usePathname();
  const [time, setTime] = useState<string>('');

  useEffect(() => {
    const updateTime = () => {
      setTime(new Date().toUTCString().slice(17, 25) + ' UTC');
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { label: 'Overview', href: '/', icon: Layers },
    { label: 'Live Vision', href: '/live', icon: Video },
    { label: 'Model Training', href: '/train', icon: Brain },
    { label: 'Memory Search', href: '/search', icon: Search },
    { label: 'Memory Timeline', href: '/timeline', icon: Clock },
    { label: 'Ask Memory', href: '/ask', icon: Sparkles },
    { label: 'Observability', href: '/observability', icon: Activity },
  ];

  return (
    <header className="site-header">
      <div className="site-header-inner">
        <div className="site-header-top">
          {/* Logo */}
          <div className="flex items-center space-x-3">
            <Link href="/" className="flex items-center space-x-2.5 group">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-teal-300 to-cyan-500 flex items-center justify-center shadow-lg shadow-teal-500/15 group-hover:scale-105 transition-transform">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <div>
                <div className="text-sm sm:text-base font-semibold tracking-tight text-slate-100">
                  Visual Memory AI
                </div>
                <div className="text-[9px] text-slate-500 font-medium tracking-[0.16em]">
                  FROM SEEING TO REMEMBERING
                </div>
              </div>
            </Link>
          </div>

          {/* Navigation Links */}
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 rounded-full border border-slate-700/70 bg-slate-900/70 px-3 py-1.5 text-[10px] font-medium tracking-wide text-slate-300">
              <span className="h-1.5 w-1.5 rounded-full bg-teal-300" />
              DASHBOARD READY
            </div>
            <div className="hidden lg:block font-mono text-[11px] text-slate-500">
              {time || '00:00:00 UTC'}
            </div>
          </div>
        </div>

        <nav className="site-nav" aria-label="Main navigation">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href || (item.href !== '/' && pathname.startsWith(`${item.href}/`));
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className="site-nav-link"
                  data-active={isActive}
                >
                  <Icon className="h-3.5 w-3.5" aria-hidden="true" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
        </nav>
      </div>
    </header>
  );
}
