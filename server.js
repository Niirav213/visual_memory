/**
 * Visual Memory AI — Express.js High-Performance Streaming & API Gateway
 * Replaces sluggish Streamlit with a heavy, ultra-fast Node/Express + React architecture.
 * Features:
 *  - Low-latency binary WebSocket frame relay for 30+ FPS smooth rendering
 *  - High-throughput Express reverse-proxy to FastAPI ML engine
 *  - Zero-lag frame broadcasting to HTML5 Canvas
 */

const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const cors = require('cors');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const server = http.createServer(app);

const PORT = process.env.PORT || 5000;
const FASTAPI_TARGET = process.env.FASTAPI_URL || 'http://127.0.0.1:8000';

app.use(cors());
app.use(express.json());

// 1. High-Performance WebSocket Streaming Server
const wss = new WebSocket.Server({ server, path: '/stream/ws' });

let backendWs = null;
let lastFrameTime = Date.now();
let serverFps = 0;
let frameCount = 0;

function connectToFastAPIWebSocket() {
  const wsUrl = `ws://127.0.0.1:8000/api/stream/ws`;
  console.log(`[Express Gateway] Connecting to FastAPI binary stream: ${wsUrl}`);
  
  try {
    backendWs = new WebSocket(wsUrl);
    backendWs.binaryType = 'arraybuffer';

    backendWs.on('open', () => {
      console.log('[Express Gateway] Successfully connected to FastAPI video stream!');
    });

    backendWs.on('message', (data) => {
      frameCount++;
      const now = Date.now();
      if (now - lastFrameTime >= 1000) {
        serverFps = frameCount;
        frameCount = 0;
        lastFrameTime = now;
      }

      // Relay binary frame directly to all connected React clients
      wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
          client.send(data, { binary: true });
        }
      });
    });

    backendWs.on('close', () => {
      console.log('[Express Gateway] FastAPI stream disconnected. Reconnecting in 2s...');
      setTimeout(connectToFastAPIWebSocket, 2000);
    });

    backendWs.on('error', (err) => {
      console.warn('[Express Gateway] Backend stream socket error:', err.message);
      backendWs.close();
    });
  } catch (err) {
    console.warn('[Express Gateway] Connection failed:', err.message);
    setTimeout(connectToFastAPIWebSocket, 3000);
  }
}

// Initial connection
connectToFastAPIWebSocket();

// Client connection tracking
wss.on('connection', (ws) => {
  console.log(`[Express Gateway] React client connected to live WebSocket. Total clients: ${wss.clients.size}`);
  
  ws.on('close', () => {
    console.log(`[Express Gateway] React client disconnected. Active clients: ${wss.clients.size}`);
  });
});

// 2. Health & Telemetry Gateway Endpoints
app.get('/gateway/status', (req, res) => {
  res.json({
    status: 'online',
    engine: 'Express.js + WebSocket Gateway',
    connected_clients: wss.clients.size,
    fps: serverFps,
    target: FASTAPI_TARGET,
    uptime_seconds: Math.floor(process.uptime()),
  });
});

// 3. Proxy all /api requests directly to high-speed FastAPI
app.use(
  '/api',
  createProxyMiddleware({
    target: FASTAPI_TARGET,
    changeOrigin: true,
    ws: true,
    logLevel: 'warn',
  })
);

// 4. Proxy media snapshots
app.use(
  '/media',
  createProxyMiddleware({
    target: FASTAPI_TARGET,
    changeOrigin: true,
    logLevel: 'warn',
  })
);

server.listen(PORT, () => {
  console.log('===========================================================');
  console.log(`  🚀 Express.js Streaming Gateway active on http://localhost:${PORT}`);
  console.log(`  ⚡ Ultra-fast React WebSocket stream: ws://localhost:${PORT}/stream/ws`);
  console.log(`  🔗 Proxied to Python ML backend on ${FASTAPI_TARGET}`);
  console.log('===========================================================');
});
