import React, { useState, useEffect, useRef } from 'react';
import logoUrl from '../assets/logo.png';
import {
  AudioLines,
  Clapperboard,
  Radio,
  ShieldCheck,
  TriangleAlert,
  Trash2,
  Settings,
  Cpu,
  RefreshCw,
  Sparkles,
} from 'lucide-react';
import { fetchLiveKitStatus, getLiveKitUrlOverride, setLiveKitUrlOverride, getGeminiApiKey, setGeminiApiKey, getGeminiModel, setGeminiModel } from '../services/livekitHttp';
import { clearGpuMemory, getApiUrl, setApiUrl, resetApiUrl, isDefaultApiUrl, testApiUrl, isLoopbackUrl, isPageOnLoopback, DEFAULT_API_URL } from '../services/api';
import { refreshBackendStatus, useBackendStatus } from '../services/telemetry';

// The pill needs a stable state while the very first poll is in flight; the
// shared poller flips `online` after one successful /health, and offline only
// after several consecutive failures (see services/telemetry.js).
export default function Header({ onToast, mode = 'studio', onModeChange }) {
  // The shared poller owns /health, /gpu and /livekit/status for the whole app.
  const {
    health,
    gpu,
    liveKit: liveKitStatus,
    online: isOnline,
    offline,
    checking,
  } = useBackendStatus();
  const [showSettings, setShowSettings] = useState(false);
  const [apiUrlInput, setApiUrlInput] = useState(getApiUrl());
  const [liveKitUrlInput, setLiveKitUrlInput] = useState(getLiveKitUrlOverride());
  const [geminiKeyInput, setGeminiKeyInput] = useState(getGeminiApiKey());
  const [geminiModelInput, setGeminiModelInput] = useState(getGeminiModel());
  const [configError, setConfigError] = useState(null);
  const [isSaving, setIsSaving] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  // Result of the explicit "Verify" probe; the periodic status may lag behind.
  const [verifyResult, setVerifyResult] = useState(null);
  const settingsRef = useRef(null);
  const settingsButtonRef = useRef(null);
  const modeButtonsRef = useRef({ studio: null, live: null });

  const activeUrl = getApiUrl();
  const trimmedInput = apiUrlInput.trim();
  const loopbackWarning =
    trimmedInput && isLoopbackUrl(trimmedInput) && !isPageOnLoopback();
  const hasOverride = !isDefaultApiUrl();
  const liveKitDetails = verifyResult ?? liveKitStatus;
  const statusPending = !isOnline && !offline;

  const checkStatus = () => refreshBackendStatus({ forceLiveKit: true }).catch(() => {});

  const closeSettings = ({ restoreFocus = true } = {}) => {
    setShowSettings(false);
    if (restoreFocus) {
      // Wait for the popover to leave the DOM before moving focus back.
      setTimeout(() => settingsButtonRef.current?.focus?.(), 0);
    }
  };

  // Escape / outside click dismiss the popover, matching every other modal in
  // the app. Focus moves into the popover on open and back to the gear on close.
  useEffect(() => {
    if (!showSettings) return undefined;
    const onKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.stopPropagation();
        closeSettings();
      }
    };
    const onPointerDown = (event) => {
      if (settingsRef.current?.contains(event.target)) return;
      if (settingsButtonRef.current?.contains(event.target)) return;
      closeSettings({ restoreFocus: false });
    };
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('touchstart', onPointerDown, { passive: true });
    const focusTimer = setTimeout(() => {
      settingsRef.current?.querySelector('input, button, select')?.focus?.();
    }, 0);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('touchstart', onPointerDown);
      clearTimeout(focusTimer);
    };
  }, [showSettings]);

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
      await testApiUrl(value);
      setApiUrl(value);
      setLiveKitUrlOverride(liveKitUrlInput);
      setGeminiApiKey(geminiKeyInput);
      setGeminiModel(geminiModelInput);
      setVerifyResult(null);
      closeSettings();
      if (onToast) onToast(`Connected to ${value}`, 'success');
      checkStatus();
    } catch (err) {
      setConfigError(err.message);
    } finally {
      setIsSaving(false);
    }
  };

  const handleSaveDirect = (e) => {
    e?.preventDefault();
    const value = trimmedInput || DEFAULT_API_URL;
    // Reject if the user pasted the full page URL (e.g. http://localhost:3000/?api=https://...)
    try {
      const u = new URL(value);
      if (u.search.includes('api=') || u.search.includes('backend=')) {
        setConfigError('That looks like a page URL, not an API base URL. Paste only the backend address (e.g. https://my-server.com).');
        return;
      }
    } catch { /* non-absolute URL — let setApiUrl handle it */ }
    setApiUrl(value);
    setLiveKitUrlOverride(liveKitUrlInput);
    setGeminiApiKey(geminiKeyInput);
    setGeminiModel(geminiModelInput);
    setVerifyResult(null);
    closeSettings();
    if (onToast) onToast(`Saved direct endpoint: ${value}`, 'info');
    checkStatus();
  };

  const openSettings = () => {
    setApiUrlInput(getApiUrl());
    setLiveKitUrlInput(getLiveKitUrlOverride());
    setGeminiKeyInput(getGeminiApiKey());
    setGeminiModelInput(getGeminiModel());
    setConfigError(null);
    setShowSettings(true);
  };

  const handleVerifyConnection = async () => {
    setIsSaving(true);
    try {
      const lk = await fetchLiveKitStatus({ verify: true });
      setVerifyResult(lk);
      const bad = (lk?.warnings || []).length;
      if (onToast) {
        onToast(
          bad === 0
            ? 'Connection checks passed — LiveKit + Gemini ready.'
            : `Connection check found ${bad} issue${bad === 1 ? '' : 's'} — see the list below.`,
          bad === 0 ? 'success' : 'error'
        );
      }
      // The backend caches the fresh probe; pull it into the shared snapshot.
      checkStatus();
    } catch (err) {
      setConfigError(err?.message || 'Verification failed.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleResetApiUrl = () => {
    const base = resetApiUrl();
    setLiveKitUrlOverride('');
    setGeminiApiKey('');
    setGeminiModel('');
    setApiUrlInput(base);
    setLiveKitUrlInput('');
    setGeminiKeyInput('');
    setGeminiModelInput('');
    setConfigError(null);
    setVerifyResult(null);
    if (onToast) onToast('Reset to default endpoint', 'info');
    checkStatus();
  };

  return (
    <header className="app-header glass-panel">
      <div className="header-brand">
        <div className="brand-logo">
          <img src={logoUrl} alt="FLOAT Logo" className="brand-logo-img" />
        </div>
        <div>
          <div className="brand-title-wrap">
            <h1 className="brand-title">FLOAT <span className="gradient-text">Studio</span></h1>
            <span className="version-badge">v5.2</span>
          </div>
          <p className="brand-subtitle">Audio-driven talking portraits</p>
        </div>
      </div>

      <nav
        className="mode-switch"
        role="tablist"
        aria-label="Workspace mode"
        onKeyDown={(event) => {
          if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
          event.preventDefault();
          const next = mode === 'studio' ? 'live' : 'studio';
          onModeChange?.(next);
          // Follow the roving tabindex so focus stays on the active tab.
          modeButtonsRef.current[next]?.focus?.();
        }}
      >
        <button
          type="button"
          role="tab"
          id="mode-tab-studio"
          aria-selected={mode === 'studio'}
          aria-controls="mode-panel-studio"
          tabIndex={mode === 'studio' ? 0 : -1}
          ref={(el) => { modeButtonsRef.current.studio = el; }}
          className={`mode-tab ${mode === 'studio' ? 'is-active' : ''}`}
          onClick={() => onModeChange?.('studio')}
        >
          <Clapperboard size={15} />
          <span>Studio</span>
        </button>
        <button
          type="button"
          role="tab"
          id="mode-tab-live"
          aria-selected={mode === 'live'}
          aria-controls="mode-panel-live"
          tabIndex={mode === 'live' ? 0 : -1}
          ref={(el) => { modeButtonsRef.current.live = el; }}
          className={`mode-tab ${mode === 'live' ? 'is-active' : ''}`}
          onClick={() => onModeChange?.('live')}
        >
          <AudioLines size={15} />
          <span>Live Audio</span>
          <span className="mode-live-dot" aria-hidden="true" />
        </button>
      </nav>

      <div className="header-actions">
        {/* Connection Status Pill. It re-renders every few seconds, so it is a
            labelled group rather than a live region: screen readers get the
            state on focus, not a new announcement on every poll. */}
        <div
          className={`status-pill ${isOnline ? 'status-online' : statusPending ? 'status-standby' : 'status-offline'}`}
          role="group"
          aria-label={`Backend ${isOnline ? 'online' : statusPending ? 'connecting' : 'offline'} at ${getApiUrl()}`}
          title={isOnline ? `Backend: ${getApiUrl()}` : statusPending ? 'Contacting the backend…' : 'Backend unreachable'}
        >
          <span className={isOnline ? 'live-dot' : 'offline-dot'} aria-hidden="true" />
          <span className="status-label" aria-hidden="true">
            {isOnline
              ? (health?.model_loaded ? 'Model ready' : 'Server online')
              : statusPending
                ? 'Connecting…'
                : 'Disconnected'}
          </span>
        </div>

        {/* LiveKit SFU Status Pill */}
        <div
          className={`status-pill ${liveKitStatus?.configured ? 'status-online' : 'status-standby'}`}
          role="group"
          aria-label={`LiveKit ${liveKitStatus?.configured ? 'configured' : 'not configured'}`}
          title={liveKitStatus?.configured ? `LiveKit SFU active (${liveKitStatus.url})` : 'LiveKit SFU standby (run docker-compose.livekit.yml)'}
        >
          <Radio size={13} aria-hidden="true" />
          <span className="status-label" aria-hidden="true">
            {liveKitStatus?.configured ? 'LiveKit' : 'LiveKit off'}
          </span>
        </div>

        {/* GPU Memory Widget */}
        {gpu?.memory && (
          <div className="gpu-widget" title={`GPU: ${gpu.info?.name || 'NVIDIA GPU'}`}>
            <Cpu size={15} className="gpu-icon" aria-hidden="true" />
            <div className="gpu-text">
              <span className="gpu-val">
                {gpu.memory.allocated_gb} / {gpu.memory.total_gb} GB
              </span>
              <div
                className="gpu-bar-bg"
                role="progressbar"
                aria-label="GPU memory in use"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(Math.min(100, gpu.memory.usage_percent || 0))}
              >
                <div
                  className="gpu-bar-fill"
                  style={{ width: `${Math.min(gpu.memory.usage_percent, 100)}%` }}
                />
              </div>
            </div>
            <button
              type="button"
              className="btn-icon-subtle"
              onClick={handleClearGpu}
              disabled={isClearing}
              title="Clear GPU VRAM cache"
              aria-label="Clear GPU VRAM cache"
            >
              <Trash2 size={14} className={isClearing ? 'animate-spin' : ''} aria-hidden="true" />
            </button>
          </div>
        )}

        {/* Refresh Status */}
        <button
          type="button"
          className="btn-icon"
          onClick={checkStatus}
          disabled={checking}
          title="Refresh server status"
          aria-label={checking ? 'Refreshing server status' : 'Refresh server status'}
          aria-busy={checking}
        >
          <RefreshCw size={16} className={checking ? 'animate-spin' : ''} aria-hidden="true" />
        </button>

        {/* Settings Toggle */}
        <button
          ref={settingsButtonRef}
          className="btn-icon"
          onClick={() => (showSettings ? closeSettings() : openSettings())}
          title="Server & network settings"
          aria-label="Server and network settings"
          aria-haspopup="dialog"
          aria-expanded={showSettings}
        >
          <Settings size={16} />
        </button>
      </div>

      {/* Settings Modal Popover */}
      {showSettings && (
        <div
          ref={settingsRef}
          className="settings-popover glass-panel"
          style={{ width: '380px' }}
          role="dialog"
          aria-modal="true"
          aria-labelledby="settings-title"
        >
          <h3 className="settings-title" id="settings-title">Server &amp; Network Configuration</h3>

          <div className="settings-active">
            <span className="settings-label">
              Active API Endpoint {hasOverride ? '(custom override)' : '(default proxy)'}
            </span>
            <code className="settings-active-url">{activeUrl}</code>
            {getLiveKitUrlOverride() && (
              <span className="settings-label" style={{ marginTop: '4px' }}>
                LiveKit Override: <code className="settings-active-url">{getLiveKitUrlOverride()}</code>
              </span>
            )}
            {getGeminiApiKey() && (
              <span className="settings-label" style={{ marginTop: '4px' }}>
                Gemini API key: <code className="settings-active-url">••••{getGeminiApiKey().slice(-4)}</code>
              </span>
            )}
            {getGeminiModel() && (
              <span className="settings-label" style={{ marginTop: '4px' }}>
                Gemini model: <code className="settings-active-url">{getGeminiModel()}</code>
              </span>
            )}
          </div>

          <form onSubmit={handleSaveApiUrl}>
            {/* Direct Presets for API */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
              <label className="settings-label" htmlFor="api-url-input" style={{ marginBottom: 0 }}>
                API Base URL (FastAPI)
              </label>
              <div style={{ display: 'flex', gap: '4px' }}>
                <button
                  type="button"
                  className="btn-secondary btn-xs"
                  onClick={() => {
                    const host = typeof window !== 'undefined' ? window.location.hostname || '127.0.0.1' : '127.0.0.1';
                    setApiUrlInput(`http://${host}:8000`);
                    setConfigError(null);
                  }}
                  title="Direct to backend server port 8000"
                >
                  Direct (:8000)
                </button>
                <button
                  type="button"
                  className="btn-secondary btn-xs"
                  onClick={() => {
                    setApiUrlInput(DEFAULT_API_URL);
                    setConfigError(null);
                  }}
                  title="Route through same-origin Vite proxy"
                >
                  Proxy (/api)
                </button>
              </div>
            </div>

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
                This page is served from {window.location.host} — <code>localhost</code> points at the visitor's machine.
              </p>
            )}

            {/* LiveKit SFU URL Section */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px', marginTop: '10px' }}>
              <label className="settings-label" htmlFor="livekit-url-input" style={{ marginBottom: 0 }}>
                LiveKit SFU URL (WebSocket)
              </label>
              <div style={{ display: 'flex', gap: '4px' }}>
                <button
                  type="button"
                  className="btn-secondary btn-xs"
                  onClick={() => {
                    const host = typeof window !== 'undefined' ? window.location.hostname || '127.0.0.1' : '127.0.0.1';
                    setLiveKitUrlInput(`ws://${host}:7880`);
                  }}
                  title="Direct to LiveKit SFU port 7880"
                >
                  Direct (:7880)
                </button>
                {liveKitUrlInput && (
                  <button
                    type="button"
                    className="btn-secondary btn-xs"
                    onClick={() => setLiveKitUrlInput('')}
                    title="Clear override to receive URL from backend"
                  >
                    Auto
                  </button>
                )}
              </div>
            </div>

            <input
              id="livekit-url-input"
              type="text"
              value={liveKitUrlInput}
              onChange={(e) => setLiveKitUrlInput(e.target.value)}
              placeholder="ws://127.0.0.1:7880 (default auto-aligned)"
              className="settings-input"
              spellCheck={false}
              autoComplete="off"
            />

            {/* Gemini API Key Section */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px', marginTop: '10px' }}>
              <label className="settings-label" htmlFor="gemini-key-input" style={{ marginBottom: 0 }}>
                Gemini API Key
              </label>
              {geminiKeyInput && (
                <button
                  type="button"
                  className="btn-secondary btn-xs"
                  onClick={() => setGeminiKeyInput('')}
                  title="Clear the stored Gemini API key"
                >
                  Clear
                </button>
              )}
            </div>

            <input
              id="gemini-key-input"
              type="password"
              value={geminiKeyInput}
              onChange={(e) => setGeminiKeyInput(e.target.value)}
              placeholder="AIza… (used by the Gemini Live engine)"
              className="settings-input"
              spellCheck={false}
              autoComplete="off"
              autoCapitalize="none"
            />
            <p className="settings-hint">
              Sent to the server with each <strong>Gemini Live</strong> session. Stored only in this
              browser; for a persistent key use <code>[gemini] &rarr; api_key</code> in{' '}
              <code>fast_float/config/secerts.toml</code> instead.
            </p>

            {/* Gemini Model Section */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px', marginTop: '10px' }}>
              <label className="settings-label" htmlFor="gemini-model-input" style={{ marginBottom: 0 }}>
                Gemini Model
              </label>
              {geminiModelInput && (
                <button
                  type="button"
                  className="btn-secondary btn-xs"
                  onClick={() => setGeminiModelInput('')}
                  title="Use the server's default (plugin) Gemini Live model"
                >
                  Auto
                </button>
              )}
            </div>

            <input
              id="gemini-model-input"
              type="text"
              value={geminiModelInput}
              onChange={(e) => setGeminiModelInput(e.target.value)}
              placeholder="gemini-2.5-flash-native-audio-preview-12-2025 (empty = plugin default)"
              className="settings-input"
              spellCheck={false}
              autoComplete="off"
              autoCapitalize="none"
            />
            <p className="settings-hint">
              Pins the Live API model for <strong>Gemini Live</strong> sessions. Empty uses the worker's
              default (server <code>[gemini] &rarr; model</code> or <code>GEMINI_MODEL</code>).
            </p>

            {/* Connection / readiness checks */}
            <div className="settings-checks">
              <div className="settings-checks-head">
                <span className="settings-label">Connection Checks</span>
                <button
                  type="button"
                  className="btn-secondary btn-xs"
                  onClick={handleVerifyConnection}
                  disabled={isSaving}
                  title="Probe the LiveKit SFU and re-run every readiness check"
                >
                  {isSaving ? 'Verifying…' : 'Verify'}
                </button>
              </div>
              {liveKitDetails?.checks?.length ? (
                <ul className="settings-checks-list">
                  {liveKitDetails.checks.map((check) => (
                    <li key={check.name} className={`settings-check settings-check--${check.status || 'skip'}`}>
                      {check.status === 'ok' ? <ShieldCheck size={13} /> : <TriangleAlert size={13} />}
                      <span>
                        <strong>{check.name.replace(/_/g, ' ')}</strong>
                        {check.message}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="settings-warning">Not available — is the backend reachable?</p>
              )}
            </div>

            {configError && <p className="settings-error">{configError}</p>}

            <p className="settings-hint">
              Use <code>Direct (:8000)</code> to talk straight to FastAPI, or <code>Proxy (/api)</code> for reverse proxying.
            </p>

            <div className="settings-buttons">
              <button
                type="button"
                className="btn-secondary btn-sm"
                disabled={isSaving}
                onClick={handleResetApiUrl}
                title="Reset all URLs to factory defaults"
              >
                Reset
              </button>
              <button
                type="button"
                className="btn-secondary btn-sm"
                disabled={isSaving}
                onClick={() => closeSettings()}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn-secondary btn-sm"
                disabled={isSaving}
                onClick={handleSaveDirect}
                title="Save URLs directly without checking connection"
              >
                Save Direct
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
