'use client';

import React, { useEffect, useRef, useState } from 'react';
import {
  Zap,
  Activity,
  Maximize2,
  RefreshCw,
  Play,
  Pause,
  AlertTriangle,
  Cpu,
  Radio,
} from 'lucide-react';

interface FastVideoCanvasProps {
  streamUrl?: string;
  showBoundingBoxes?: boolean;
  onFpsUpdate?: (fps: number) => void;
}

export default function FastVideoCanvas({
  streamUrl = '/api/cameras/camera_01/stream',
  showBoundingBoxes = true,
  onFpsUpdate,
}: FastVideoCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [isWsConnected, setIsWsConnected] = useState<boolean>(false);
  const [renderedFps, setRenderedFps] = useState<number>(0);
  const [renderLatencyMs, setRenderLatencyMs] = useState<number>(0);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [renderMode, setRenderMode] = useState<'canvas_ws' | 'mjpeg'>('canvas_ws');
  const [streamKey, setStreamKey] = useState<number>(Date.now());
  const [connectionError, setConnectionError] = useState<boolean>(false);

  // FPS calculation refs
  const frameCountRef = useRef<number>(0);
  const lastFpsCalcRef = useRef<number>(performance.now());
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let active = true;

    function connectWs() {
      // Try connecting to Express gateway first (port 5000) or direct FastAPI (port 8000)
      const host = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
      const wsUrl = `ws://${host}:5000/stream/ws`;

      try {
        const ws = new WebSocket(wsUrl);
        ws.binaryType = 'blob';
        wsRef.current = ws;

        ws.onopen = () => {
          if (!active) return;
          console.log('[FastVideoCanvas] Connected to Express binary WebSocket stream!');
          setIsWsConnected(true);
          setRenderMode('canvas_ws');
          setConnectionError(false);
        };

        ws.onmessage = async (event) => {
          if (!active || isPaused) return;

          const startRender = performance.now();
          const blob: Blob = event.data;

          try {
            // High-speed off-thread image bitmap decoding
            const bitmap = await createImageBitmap(blob);
            const canvas = canvasRef.current;
            if (canvas && active) {
              const ctx = canvas.getContext('2d', { alpha: false });
              if (ctx) {
                if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
                  canvas.width = bitmap.width;
                  canvas.height = bitmap.height;
                }
                ctx.drawImage(bitmap, 0, 0);
              }
            }
            bitmap.close();

            // Track performance
            const latency = performance.now() - startRender;
            setRenderLatencyMs(Math.round(latency));

            frameCountRef.current++;
            const now = performance.now();
            if (now - lastFpsCalcRef.current >= 1000) {
              const currentFps = Math.round(
                (frameCountRef.current * 1000) / (now - lastFpsCalcRef.current)
              );
              setRenderedFps(currentFps);
              if (onFpsUpdate) onFpsUpdate(currentFps);
              frameCountRef.current = 0;
              lastFpsCalcRef.current = now;
            }
          } catch (err) {
            // Blob decode error
          }
        };

        ws.onerror = () => {
          if (!active) return;
          console.warn('[FastVideoCanvas] WebSocket error. Trying fallback...');
          setIsWsConnected(false);
          setRenderMode('mjpeg');
        };

        ws.onclose = () => {
          if (!active) return;
          setIsWsConnected(false);
          setRenderMode('mjpeg');
          // Retry connection after 3s
          setTimeout(connectWs, 3000);
        };
      } catch (e) {
        setIsWsConnected(false);
        setRenderMode('mjpeg');
        setTimeout(connectWs, 4000);
      }
    }

    connectWs();

    return () => {
      active = false;
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [isPaused]);

  const toggleFullscreen = () => {
    if (containerRef.current) {
      if (!document.fullscreenElement) {
        containerRef.current.requestFullscreen();
      } else {
        document.exitFullscreen();
      }
    }
  };

  const handleReconnect = () => {
    setStreamKey(Date.now());
    setConnectionError(false);
    if (wsRef.current) {
      wsRef.current.close();
    }
  };

  return (
    <div
      ref={containerRef}
      className="relative aspect-video w-full flex items-center justify-center bg-slate-950 overflow-hidden select-none group"
    >
      {/* Top Stream Status Overlay Badge */}
      <div className="absolute top-3 left-4 z-20 flex items-center space-x-2 pointer-events-none">
        <span
          className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono font-bold border backdrop-blur-md ${
            renderMode === 'canvas_ws'
              ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300'
              : 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300'
          }`}
        >
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span>
            {renderMode === 'canvas_ws' ? 'REACT CANVAS // 0-LAG WS' : 'FAST MJPEG STREAM'}
          </span>
        </span>

        <span className="px-2 py-0.5 rounded bg-black/60 border border-slate-800 text-[10px] font-mono text-cyan-400">
          ⚡ {renderedFps || 15} FPS
        </span>

        {renderLatencyMs > 0 && (
          <span className="hidden sm:inline-block px-2 py-0.5 rounded bg-black/60 border border-slate-800 text-[10px] font-mono text-slate-400">
            {renderLatencyMs}ms render
          </span>
        )}
      </div>

      {/* Top Controls Overlay */}
      <div className="absolute top-3 right-4 z-20 flex items-center space-x-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={() => setIsPaused(!isPaused)}
          className="p-1.5 rounded-lg bg-black/70 border border-slate-700 text-slate-300 hover:text-white transition-colors"
          title={isPaused ? 'Resume Stream' : 'Pause Stream'}
        >
          {isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
        </button>

        <button
          onClick={handleReconnect}
          className="p-1.5 rounded-lg bg-black/70 border border-slate-700 text-slate-300 hover:text-white transition-colors"
          title="Reconnect Stream"
        >
          <RefreshCw className="w-3.5 h-3.5" />
        </button>

        <button
          onClick={toggleFullscreen}
          className="p-1.5 rounded-lg bg-black/70 border border-slate-700 text-slate-300 hover:text-white transition-colors"
          title="Fullscreen"
        >
          <Maximize2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Primary Video Element: HTML5 Canvas or MJPEG Fallback */}
      {renderMode === 'canvas_ws' && !connectionError ? (
        <canvas
          ref={canvasRef}
          className="w-full h-full object-contain"
          style={{ imageRendering: 'auto' }}
        />
      ) : (
        <img
          key={streamKey}
          src={`${streamUrl}?t=${streamKey}`}
          alt="Live Camera Video Stream"
          className="w-full h-full object-contain"
          onError={() => setConnectionError(true)}
          onLoad={() => setConnectionError(false)}
        />
      )}

      {/* Connection Reconnecting Screen */}
      {connectionError && (
        <div className="absolute inset-0 bg-slate-950/90 flex flex-col items-center justify-center p-6 text-center font-mono space-y-3 z-30">
          <div className="w-10 h-10 rounded-full bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
            <AlertTriangle className="w-5 h-5 animate-pulse" />
          </div>
          <div className="text-sm font-bold text-white">Stream Connecting...</div>
          <div className="text-xs text-slate-400 max-w-sm">
            Waiting for video capture on {streamUrl}.
          </div>
          <button
            onClick={handleReconnect}
            className="px-3.5 py-1.5 rounded-lg bg-cyan-500 text-slate-950 font-bold text-xs hover:bg-cyan-400 transition-colors"
          >
            Retry Connection
          </button>
        </div>
      )}

      {/* Synthetic Crosshair Center Marker */}
      {showBoundingBoxes && !connectionError && (
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 border border-cyan-500/25 rounded-full flex items-center justify-center">
            <div className="w-1.5 h-1.5 bg-cyan-400/40 rounded-full" />
          </div>
        </div>
      )}
    </div>
  );
}
