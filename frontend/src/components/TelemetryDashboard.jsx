import React, { useState, useEffect } from 'react';
import {
  Activity,
  Cpu,
  Zap,
  Clock,
  Timer,
  Layers,
  Download,
  RotateCw
} from 'lucide-react';
import { fetchGpuMetrics, fetchHealth } from '../services/api';

export default function TelemetryDashboard({ latestResult, history = [] }) {
  const [gpu, setGpu] = useState(null);
  const [health, setHealth] = useState(null);

  const refreshTelemetry = async () => {
    try {
      const g = await fetchGpuMetrics();
      setGpu(g);
    } catch (e) {
      // GPU info might not be available if no CUDA
    }
    try {
      const h = await fetchHealth();
      setHealth(h);
    } catch (e) {
      // Health check
    }
  };

  useEffect(() => {
    refreshTelemetry();
    const interval = setInterval(refreshTelemetry, 5000);
    return () => clearInterval(interval);
  }, []);

  const perf = latestResult?.performance || {
    preprocess_time: 0,
    inference_time: 0,
    total_time: 0,
    fps: 0,
  };

  return (
    <div className="telemetry-section glass-panel">
      <div className="section-header">
        <div className="section-title-wrap">
          <Activity size={18} className="section-icon text-cyan" />
          <h2 className="section-title">Telemetry & Performance Hub</h2>
        </div>
        <button className="btn-icon-subtle" onClick={refreshTelemetry} title="Refresh Telemetry">
          <RotateCw size={15} />
        </button>
      </div>

      {/* Grid of Key Performance Indicators */}
      <div className="kpi-grid">
        {/* Metric 1: FPS */}
        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-title">Inference Speed</span>
            <Zap size={16} className="text-cyan" />
          </div>
          <div className="kpi-value-row">
            <span className="kpi-value gradient-text">{perf.fps > 0 ? perf.fps : '--'}</span>
            <span className="kpi-unit">FPS</span>
          </div>
          <span className="kpi-hint">
            {health?.performance_stats?.average_fps ? `Avg: ${health.performance_stats.average_fps} FPS` : 'Flow Matching ODE'}
          </span>
        </div>

        {/* Metric 2: Total Pipeline Time */}
        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-title">Total Latency</span>
            <Clock size={16} className="text-indigo" />
          </div>
          <div className="kpi-value-row">
            <span className="kpi-value">{perf.total_time > 0 ? perf.total_time : '--'}</span>
            <span className="kpi-unit">sec</span>
          </div>
          <span className="kpi-hint">End-to-end request turnaround</span>
        </div>

        {/* Metric 3: Model ODE Inference Time */}
        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-title">Model Inference</span>
            <Timer size={16} className="text-cyan" />
          </div>
          <div className="kpi-value-row">
            <span className="kpi-value">{perf.inference_time > 0 ? perf.inference_time : '--'}</span>
            <span className="kpi-unit">sec</span>
          </div>
          <span className="kpi-hint">Wav2Vec + Transformer runtime</span>
        </div>

        {/* Metric 4: Preprocess Time */}
        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-title">Prep & Audio FFT</span>
            <Layers size={16} className="text-indigo" />
          </div>
          <div className="kpi-value-row">
            <span className="kpi-value">{perf.preprocess_time > 0 ? perf.preprocess_time : '--'}</span>
            <span className="kpi-unit">sec</span>
          </div>
          <span className="kpi-hint">Audio normalizer & alignment</span>
        </div>
      </div>

      {/* GPU Hardware Diagnostics Bar */}
      {gpu?.info && (
        <div className="gpu-telemetry-box">
          <div className="gpu-telemetry-header">
            <div className="gpu-telemetry-title">
              <Cpu size={16} className="text-cyan" />
              <span>{gpu.info.name || 'GPU Hardware Accelerator'}</span>
            </div>
            {gpu.memory && (
              <span className="gpu-vram-text">
                {gpu.memory.allocated_gb} GB / {gpu.memory.total_gb} GB ({gpu.memory.usage_percent}%)
              </span>
            )}
          </div>

          {gpu.memory && (
            <div className="gpu-vram-track">
              <div
                className="gpu-vram-fill"
                style={{ width: `${Math.min(gpu.memory.usage_percent, 100)}%` }}
              />
            </div>
          )}
        </div>
      )}

      {/* History of Recent Generations */}
      {history.length > 0 && (
        <div className="history-drawer">
          <h3 className="history-title">Session Generations ({history.length})</h3>
          <div className="history-list">
            {history.map((item, idx) => (
              <div key={item.request_id || idx} className="history-row">
                <div className="history-id">
                  <span className="history-idx">#{history.length - idx}</span>
                  <code>{item.request_id}</code>
                </div>
                <div className="history-stats">
                  <span>{item.performance?.fps || 0} FPS</span>
                  <span>•</span>
                  <span>{item.performance?.total_time || 0}s</span>
                  <span>•</span>
                  <span>{item.file_size_mb || 0} MB</span>
                </div>
                <a
                  href={item.download_url}
                  download
                  className="btn-icon-subtle"
                  title="Download Video"
                >
                  <Download size={15} />
                </a>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
