import React, { useState, useEffect, useRef } from 'react';
import {
  Sparkles,
  Radio,
  Trash2,
  Settings,
  Cpu,
  RefreshCw
} from 'lucide-react';
import { fetchLiveKitStatus } from '../services/livekit';
import { fetchHealth, fetchGpuMetrics, clearGpuMemory, getApiUrl, setApiUrl, resetApiUrl, isDefaultApiUrl, testApiUrl, isLoopbackUrl, isPageOnLoopback, DEFAULT_API_URL } from '../services/api';

// Consecutive failed polls before the pill flips to Disconnected. A single
// failed poll (e.g. one slow tick while a big upload/generation is in
// flight) must not flap the UI offline.
const OFFLINE_AFTER_FAILURES = 3;

export default function Header({ onToast }) {
  const [health, setHealth] = useState(null);
  const [gpu, setGpu] = useState(null);
  const [isOnline, setIsOnline] = useState(false);
  const [liveKitStatus, setLiveKitStatus] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [apiUrlInput, setApiUrlInput] = useState(getApiUrl());
  const [configError, setConfigError] = useState(null);
  const [isSaving, setIsSaving] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const failuresRef = useRef(0);

  const activeUrl = getApiUrl();
  const trimmedInput = apiUrlInput.trim();
  const loopbackWarning =
    trimmedInput && isLoopbackUrl(trimmedInput) && !isPageOnLoopback();
  const hasOverride = !isDefaultApiUrl();

  const checkStatus = async () => {
    try {
      const hData = await fetchHealth();
      fetchLiveKitStatus().then((lk) => setLiveKitStatus(lk)).catch(() => setLiveKitStatus({ configured: false }));
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

  const handleSaveApiUrl = async (e) => {
    e.preventDefault();
    const value = trimmedInput;
    if (!value) {
      setConfigError('Enter a URL, or press Reset to use the default.');
      return;
    }
    setIsSaving(true);
    setConfigError(null);
    try {
      // Verify before committing so a typo can't strand the whole studio.
      const hData = await testApiUrl(value);
      setApiUrl(value);
      failuresRef.current = 0;
      if (hData && typeof hData === 'object') setHealth(hData);
      setIsOnline(true);
      setShowSettings(false);
      if (onToast) onToast(`Connected to ${value}`, 'success');
      checkStatus();
    } catch (err) {
      setConfigError(err.message);
    } finally {
      setIsSaving(false);
    }
  };

  const openSettings = () => {
    setApiUrlInput(getApiUrl());
    setConfigError(null);
    setShowSettings(true);
  };

  const handleResetApiUrl = () => {
    const base = resetApiUrl();
    setApiUrlInput(base);
    setConfigError(null);
    failuresRef.current = 0;
    if (onToast) onToast('Reset to default endpoint', 'info');
    checkStatus();
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

        {/* LiveKit SFU Status Pill */}
        <div
          className={`status-pill ${liveKitStatus?.configured ? 'status-online' : 'status-offline'}`}
          title={liveKitStatus?.configured ? `LiveKit SFU Active (${liveKitStatus.url})` : 'LiveKit SFU Standby (run docker-compose.livekit.yml)'}
        >
          <Radio size={13} style={{ marginRight: '5px' }} />
          <span className="status-label">
            {liveKitStatus?.configured ? 'LiveKit Ready' : 'LiveKit Standby'}
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
          onClick={openSettings}
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

          <div className="settings-active">
            <span className="settings-label">
              Active endpoint {hasOverride ? '(custom override)' : '(default proxy)'}
            </span>
            <code className="settings-active-url">{activeUrl}</code>
          </div>

          <form onSubmit={handleSaveApiUrl}>
            <label className="settings-label" htmlFor="api-url-input">API base URL</label>
            <input
              id="api-url-input"
              type="text"
              value={apiUrlInput}
              onChange={(e) => { setApiUrlInput(e.target.value); setConfigError(null); }}
              placeholder={DEFAULT_API_URL}
              className="settings-input"
              spellCheck={false}
              autoComplete="off"
            />

            {loopbackWarning && (
              <p className="settings-warning">
                This page is served from {window.location.host} — <code>localhost</code> there
                points at the visitor's own machine, so this endpoint may be unreachable remotely.
              </p>
            )}
            {configError && <p className="settings-error">{configError}</p>}

            <p className="settings-hint">
              Leave as the default to route through the same-origin <code>/api</code> proxy.
              Set a full URL only to talk to the FastAPI server directly.
            </p>

            <div className="settings-buttons">
              <button
                type="button"
                className="btn-secondary btn-sm"
                disabled={isSaving}
                onClick={handleResetApiUrl}
              >
                Reset
              </button>
              <button
                type="button"
                className="btn-secondary btn-sm"
                disabled={isSaving}
                onClick={() => setShowSettings(false)}
              >
                Cancel
              </button>
              <button type="submit" className="btn-primary btn-sm" disabled={isSaving}>
                {isSaving ? 'Testing…' : 'Test & Save'}
              </button>
            </div>
          </form>
        </div>
      )}
    </header>
  );
}
