import React, { useState, useEffect, useRef } from 'react';
import {
  Sparkles,
  Trash2,
  Settings,
  Cpu,
  RefreshCw
} from 'lucide-react';
import { fetchHealth, fetchGpuMetrics, clearGpuMemory, getApiUrl, setApiUrl, DEFAULT_API_URL } from '../services/api';

// Consecutive failed polls before the pill flips to Disconnected. A single
// failed poll (e.g. one slow tick while a big upload/generation is in
// flight) must not flap the UI offline.
const OFFLINE_AFTER_FAILURES = 3;

export default function Header({ onToast }) {
  const [health, setHealth] = useState(null);
  const [gpu, setGpu] = useState(null);
  const [isOnline, setIsOnline] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [apiUrlInput, setApiUrlInput] = useState(getApiUrl());
  const [isClearing, setIsClearing] = useState(false);
  const failuresRef = useRef(0);

  const checkStatus = async () => {
    try {
      const hData = await fetchHealth();
      setHealth(hData);
      failuresRef.current = 0;
      setIsOnline(true);
      try {
        const gData = await fetchGpuMetrics();
        setGpu(gData);
      } catch (e) {
        // GPU endpoint optional or no GPU
      }
    } catch (err) {
      failuresRef.current += 1;
      console.warn(`[health] poll failed (${failuresRef.current}/${OFFLINE_AFTER_FAILURES}): ${err.message}`);
      if (failuresRef.current >= OFFLINE_AFTER_FAILURES) {
        setIsOnline((prev) => {
          if (prev && onToast) onToast(`Backend unreachable at ${getApiUrl()} — check connection or Config.`, 'error');
          return false;
        });
      }
    }
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 6000);
    return () => clearInterval(interval);
  }, []);

  const handleClearGpu = async () => {
    setIsClearing(true);
    try {
      await clearGpuMemory();
      if (onToast) onToast('GPU cache cleared successfully!', 'success');
      await checkStatus();
    } catch (err) {
      if (onToast) onToast(`Failed to clear GPU: ${err.message}`, 'error');
    } finally {
      setIsClearing(false);
    }
  };

  const handleSaveApiUrl = (e) => {
    e.preventDefault();
    setApiUrl(apiUrlInput);
    failuresRef.current = 0;
    setShowSettings(false);
    checkStatus();
    if (onToast) onToast(`API URL updated to ${apiUrlInput}`, 'info');
  };

  return (
    <header className="header-container glass-panel">
      <div className="header-brand">
        <div className="brand-logo">
          <Sparkles className="brand-icon" size={22} />
        </div>
        <div>
          <div className="brand-title-wrap">
            <h1 className="brand-title">FLOAT <span className="gradient-text">Studio</span></h1>
            <span className="version-badge">v5.1 NVENC</span>
          </div>
          <p className="brand-subtitle">Fast Flow Matching Talking Portrait Generator</p>
        </div>
      </div>

      <div className="header-actions">
        {/* Connection Status Pill */}
        <div className={`status-pill ${isOnline ? 'status-online' : 'status-offline'}`}>
          <span className={isOnline ? 'live-dot' : 'offline-dot'} />
          <span className="status-label">
            {isOnline 
              ? (health?.model_loaded ? 'Model Ready' : 'Server Online') 
              : 'Disconnected'}
          </span>
        </div>

        {/* GPU Memory Widget */}
        {gpu?.memory && (
          <div className="gpu-widget" title={`GPU: ${gpu.info?.name || 'NVIDIA GPU'}`}>
            <Cpu size={16} className="gpu-icon" />
            <div className="gpu-text">
              <span className="gpu-val">
                {gpu.memory.allocated_gb} / {gpu.memory.total_gb} GB
              </span>
              <div className="gpu-bar-bg">
                <div 
                  className="gpu-bar-fill" 
                  style={{ width: `${Math.min(gpu.memory.usage_percent, 100)}%` }} 
                />
              </div>
            </div>
            <button 
              className="btn-icon" 
              onClick={handleClearGpu} 
              disabled={isClearing} 
              title="Clear GPU VRAM cache"
            >
              <Trash2 size={14} className={isClearing ? 'animate-spin' : ''} />
            </button>
          </div>
        )}

        {/* Refresh Status */}
        <button className="btn-icon" onClick={checkStatus} title="Refresh Server Status">
          <RefreshCw size={16} />
        </button>

        {/* Settings Toggle */}
        <button 
          className="btn-secondary btn-sm" 
          onClick={() => setShowSettings(!showSettings)}
          title="Configure API Endpoint"
        >
          <Settings size={15} />
          <span>Config</span>
        </button>
      </div>

      {/* Settings Modal Popover */}
      {showSettings && (
        <div className="settings-popover glass-panel">
          <h3 className="settings-title">Backend Connection</h3>
          <form onSubmit={handleSaveApiUrl}>
            <label className="settings-label">FastAPI Server URL</label>
            <input 
              type="text" 
              value={apiUrlInput} 
              onChange={(e) => setApiUrlInput(e.target.value)} 
              placeholder="http://localhost:8000"
              className="settings-input"
            />
            <div className="settings-buttons">
              <button
                type="button"
                className="btn-secondary btn-sm"
                title={`Reset to default (${DEFAULT_API_URL})`}
                onClick={() => setApiUrlInput(DEFAULT_API_URL)}
              >
                Reset
              </button>
              <button type="button" className="btn-secondary btn-sm" onClick={() => setShowSettings(false)}>
                Cancel
              </button>
              <button type="submit" className="btn-primary btn-sm">
                Save & Connect
              </button>
            </div>
          </form>
        </div>
      )}
    </header>
  );
}
