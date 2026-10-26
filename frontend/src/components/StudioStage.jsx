import React, { useCallback, useRef, useState, useEffect } from 'react';
import {
  Download,
  Film,
  Gauge,
  Image as ImageIcon,
  Link2,
  Play,
  Radio,
  RefreshCw,
  Sparkles,
  TriangleAlert,
  Wand2,
  Zap,
} from 'lucide-react';
import { getApiUrl, setIceServers } from '../services/api';
import { copyText } from '../services/clipboard';
import { useBackendStatus } from '../services/telemetry';
import LivePreview from './LivePreview';

/** Friendly copy for the backend's machine-readable progress stages. */
const STAGE_LABELS = {
  queued: 'Waiting for a free GPU slot',
  preprocessing: 'Preprocessing audio & portrait',
  generating: 'Solving the flow ODE',
  merging: 'Merging chunks',
  done: 'Finalising video',
  error: 'Generation failed',
};

function formatElapsed(seconds) {
  const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  const mins = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  return `${mins}:${String(secs).padStart(2, '0')}`;
}

function formatClipLength(seconds) {
  const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  if (!safe) return null;
  return safe < 60 ? `${safe.toFixed(1)}s clip` : `${formatElapsed(safe)} clip`;
}

const TRANSPORTS = [
  { id: 'auto', label: 'Auto (recommended)' },
  { id: 'livekit', label: 'LiveKit (live preview)' },
  { id: 'webrtc', label: 'WebRTC (live preview)' },
  { id: 'ws', label: 'WebSocket (live frames)' },
  { id: 'mp4', label: 'File only (MP4)' },
];

export default function StudioStage({
  onGenerate,
  onRetry,
  isGenerating,
  progress,
  generationError = null,
  generationResult,
  hasAudio,
  onToast,
  transport = 'auto',
  setTransport,
  liveRequestId,
  imagePreview,
}) {
  const [isDownloading, setIsDownloading] = useState(false);
  const [endedLiveRequestId, setEndedLiveRequestId] = useState(null);
  const [clockNow, setClockNow] = useState(() => Date.now());
  const { liveKit: liveKitStatus } = useBackendStatus();
  // Derived, not stored: the shared poller owns the status, so a second copy
  // in state can only go stale.
  const liveKitAvailable = liveKitStatus ? !!liveKitStatus.configured : null;

  const onToastRef = useRef(onToast);
  useEffect(() => {
    onToastRef.current = onToast;
  }, [onToast]);

  // Live elapsed readout while a render is running. The backend only writes a
  // progress record at stage boundaries (single-pass writes once), so the
  // clock is driven locally from its queue timestamp instead of waiting for
  // the next poll.
  useEffect(() => {
    if (!isGenerating) return undefined;
    setClockNow(Date.now());
    const timer = setInterval(() => setClockNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [isGenerating]);

  // Feed the deployment's ICE list to the raw WebRTC transport as soon as the
  // shared poller publishes it.
  useEffect(() => {
    if (!liveKitStatus) return;
    setIceServers(
      liveKitStatus.ice_servers?.length ? liveKitStatus.ice_servers : liveKitStatus.stun_urls
    );
  }, [liveKitStatus]);

  // LiveKit is the primary SFU. When it is unconfigured or unreachable, switch
  // to raw WebRTC (signalled by this API host), not straight to the WebSocket.
  useEffect(() => {
    if (liveKitAvailable === false && transport === 'livekit') {
      setTransport?.('webrtc');
      onToastRef.current?.('LiveKit is unavailable — switching to WebRTC.', 'error');
    }
  }, [liveKitAvailable, transport, setTransport]);

  // Picking the MP4 transport is an explicit "leave the live view" action.
  const handleTransportChange = (next) => {
    if (next === 'mp4') setEndedLiveRequestId(liveRequestId ?? null);
    setTransport?.(next);
  };

  const videoUrl = generationResult?.stream_url
    || (generationResult ? `${getApiUrl()}/stream/${generationResult.request_id}` : null);

  const downloadUrl = generationResult?.download_url
    || (generationResult ? `${getApiUrl()}/download/${generationResult.request_id}` : null);
  const downloadBackupUrl = generationResult?.download_backup || null;

  const handleCopyLink = async () => {
    const link = downloadUrl || downloadBackupUrl;
    if (!link) return;
    const copied = await copyText(link);
    onToast?.(
      copied
        ? 'Download link copied to clipboard.'
        : 'Could not copy automatically — the link is in the address bar or the download button.',
      copied ? 'info' : 'error'
    );
  };

  // Cross-origin <a download> is ignored by browsers, so fetch it as a blob.
  // The backend exposes a second URL (`download_backup`, usually the same file
  // via a different host) for split-horizon deployments where the primary
  // hostname is unreachable from the browser.
  const handleDownload = async () => {
    const candidates = [downloadUrl, downloadBackupUrl]
      .filter(Boolean)
      .filter((url, index, list) => list.indexOf(url) === index);
    if (!candidates.length || !generationResult?.request_id || isDownloading) return;

    setIsDownloading(true);
    let lastError = null;
    for (const url of candidates) {
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Download failed: ${res.status} ${res.statusText}`.trim());
        const blob = await res.blob();
        if (!blob.size) throw new Error('The server returned an empty file.');
        const objectUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = objectUrl;
        a.download = `float_talking_portrait_${generationResult.request_id}.mp4`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(objectUrl), 5000);
        onToast?.('Download started.', 'success');
        setIsDownloading(false);
        return;
      } catch (err) {
        lastError = err;
        console.warn('Download attempt failed', url, err);
      }
    }

    // Last resort: let the browser navigate to the file. A cross-origin
    // `download` attribute is ignored, but the tab still shows the video.
    const fallback = candidates[0];
    try {
      window.open(fallback, '_blank', 'noopener');
      onToast?.('Direct download failed — opened the file in a new tab.', 'info');
    } catch {
      onToast?.(`Download failed: ${lastError?.message || 'the file is unavailable'}.`, 'error');
    }
    setIsDownloading(false);
  };

  // Keep the live preview mounted after the POST returns: a real-time transport
  // is still playing frames buffered during the fast render.
  const livePreviewRequestId = (() => {
    if (!liveRequestId) return null;
    if (isGenerating) return liveRequestId;
    if (endedLiveRequestId === liveRequestId) return null;
    if (generationResult?.request_id !== liveRequestId) return null;
    return liveRequestId;
  })();

  const activeTransport =
    transport === 'auto'
      ? (liveKitAvailable === false ? 'webrtc' : 'livekit')
      : transport;

  // Degrade one step at a time: LiveKit → raw WebRTC → frames over WS → MP4.
  const handleLiveFallback = useCallback(
    (reason) => {
      const source = transport === 'auto' ? activeTransport : transport;
      const next = source === 'livekit' ? 'webrtc' : source === 'webrtc' ? 'ws' : 'mp4';
      const detail = reason ? ` (${reason})` : '';
      if (next === 'mp4') setEndedLiveRequestId(liveRequestId ?? null);
      setTransport?.(next);
      const message = next === 'webrtc'
        ? `LiveKit failed${detail} — switched to WebRTC.`
        : next === 'ws'
          ? `WebRTC failed${detail} — switched to low-latency frames.`
          : `Live preview unavailable${detail} — the MP4 will appear when the render finishes.`;
      onToastRef.current?.(message, 'error');
    },
    [activeTransport, liveRequestId, transport, setTransport]
  );

  // Real progress only exists for chunked renders. A single-pass render keeps
  // percent pinned at 5.0 for its whole inference, so showing a 5% bar would
  // look frozen; use an indeterminate bar + live label instead.
  const totalChunks = Number(progress?.total_chunks) || 0;
  const completedChunks = Number(progress?.completed_chunks) || 0;
  const percent = Math.max(0, Math.min(100, Number(progress?.percent) || 0));
  const isMerging = progress?.status === 'merging';
  const isSettling = progress?.status === 'done';
  const determinate = Boolean(
    isGenerating && (totalChunks > 1 || isMerging || isSettling)
  );
  const progressPercent = determinate ? percent : 0;

  const audioSeconds = Number(progress?.audio_seconds) || 0;
  const startedAtMs = Number(progress?.started_at)
    ? Number(progress.started_at) * 1000
    : Number(progress?.elapsed_s) > 0
      ? clockNow - Number(progress.elapsed_s) * 1000
      : null;
  const elapsedSeconds = startedAtMs
    ? Math.max(0, (clockNow - startedAtMs) / 1000)
    : Number(progress?.elapsed_s) || 0;

  const progressLabel = isSettling
    ? STAGE_LABELS.done
    : (progress?.message?.trim()
      || STAGE_LABELS[progress?.stage]
      || 'Generating…');

  const showLivePreview = livePreviewRequestId && activeTransport !== 'mp4';
  // A previous take must not look like the output of the render in flight.
  const showResult = !isGenerating && generationResult;
  const clipLabel = formatClipLength(audioSeconds);
  const metaBits = [
    totalChunks > 1 ? `${completedChunks}/${totalChunks} chunks` : null,
    clipLabel,
    determinate ? `${formatElapsed(elapsedSeconds)} elapsed` : null,
  ].filter(Boolean);

  return (
    <section className="glass-panel stage-panel">
      <div className="stage-head">
        <div className="stage-heading">
          <span className="stage-eyebrow">
            <Wand2 size={13} /> Output
          </span>
          <h2 className="stage-title">Studio canvas</h2>
          <p className="stage-desc">
            {isGenerating
              ? 'Frames appear here as soon as they are decoded.'
              : generationResult
                ? 'Your talking portrait is ready. Download it or generate another take.'
                : 'Your talking portrait will appear here.'}
          </p>
        </div>

        <button
          type="button"
          className="btn-primary btn-generate"
          onClick={onGenerate}
          disabled={isGenerating || !hasAudio}
          title={!hasAudio ? 'Add voice audio first' : 'Generate the talking portrait'}
        >
          {isGenerating ? (
            <>
              <span className="spinner-dot animate-spin" />
              <span>Generating…</span>
            </>
          ) : (
            <>
              <Sparkles size={18} />
              <span>{generationResult ? 'Generate another' : 'Generate video'}</span>
            </>
          )}
        </button>
      </div>

      {!hasAudio && !generationResult && (
        <div className="stage-hint">
          <span className="stage-hint-dot" />
          Record or upload a voice clip to unlock generation.
        </div>
      )}

      {generationError && !isGenerating && (
        <div className="stage-error" role="alert">
          <TriangleAlert size={16} className="stage-error-icon" />
          <div className="stage-error-body">
            <strong>Generation failed</strong>
            <span>{generationError}</span>
          </div>
          {onRetry && (
            <button
              type="button"
              className="btn-secondary btn-sm"
              onClick={onRetry}
              disabled={!hasAudio}
              title={!hasAudio ? 'Add voice audio first' : 'Try this render again'}
            >
              <RefreshCw size={14} />
              Retry
            </button>
          )}
        </div>
      )}

      <div className={`stage-viewport ${isGenerating ? 'is-generating' : ''}`}>
        {showLivePreview ? (
          <LivePreview
            requestId={livePreviewRequestId}
            transport={activeTransport}
            onStatus={(message) => onToast?.(message || 'Live stream ended', 'error')}
            onFallback={handleLiveFallback}
            onEnded={(id) => setEndedLiveRequestId(id ?? livePreviewRequestId)}
          />
        ) : showResult && videoUrl ? (
          <video
            key={videoUrl}
            src={videoUrl}
            poster={imagePreview || undefined}
            controls
            autoPlay
            loop
            playsInline
            className="stage-video"
            onError={() => onToast?.('The rendered video could not be loaded. Try downloading the MP4.', 'error')}
          />
        ) : (
          <div className="stage-empty">
            <div
              className={`stage-empty-portrait ${imagePreview ? 'has-image' : ''}`}
              style={imagePreview ? { backgroundImage: `url(${imagePreview})` } : undefined}
            >
              {!imagePreview && <ImageIcon size={30} />}
              <span className="stage-empty-ring" />
            </div>
            <div className="stage-empty-copy">
              <span className="stage-empty-badge">
                <Play size={11} /> {isGenerating ? 'Rendering' : 'Ready'}
              </span>
              <h3>{isGenerating ? 'Rendering your portrait…' : 'Bring a portrait to life'}</h3>
              <p>
                {isGenerating
                  ? 'The finished MP4 will appear here when the render completes.'
                  : (
                    <>
                      Add a voice clip, then press <strong>Generate video</strong>. With a live
                      transport selected, frames start playing before the MP4 is finished.
                    </>
                  )}
              </p>
            </div>
          </div>
        )}

        {isGenerating && (
          <div className="stage-progress">
            <div className="stage-progress-top">
              <span className="stage-progress-label">
                <span className="stage-progress-pulse" />
                <span aria-live="polite">{progressLabel}</span>
              </span>
              <span className="stage-progress-value">
                {determinate
                  ? `${Math.round(progressPercent)}%`
                  : formatElapsed(elapsedSeconds)}
              </span>
            </div>
            <div
              className={`progress-track ${determinate ? '' : 'is-indeterminate'}`}
              role="progressbar"
              aria-label="Generation progress"
              aria-valuemin={0}
              aria-valuemax={100}
              {...(determinate
                ? {
                  'aria-valuenow': Math.round(progressPercent),
                  'aria-valuetext': `${Math.round(progressPercent)}% — ${progressLabel}`,
                }
                : { 'aria-valuetext': progressLabel })}
            >
              <div
                className="progress-fill-bar"
                style={determinate ? { width: `${progressPercent}%` } : undefined}
              />
            </div>
            {metaBits.length > 0 && (
              <div className="stage-progress-meta">
                {metaBits.map((bit) => <span key={bit}>{bit}</span>)}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="stage-toolbar">
        <div className="transport-control">
          <label htmlFor="transport-select" className="transport-label">
            <Radio size={13} />
            Preview
          </label>
          <select
            id="transport-select"
            className="transport-select"
            value={transport}
            onChange={(e) => handleTransportChange(e.target.value)}
            disabled={isGenerating}
            title="How the render is previewed"
          >
            {TRANSPORTS.map((opt) => (
              <option key={opt.id} value={opt.id}>{opt.label}</option>
            ))}
          </select>
        </div>

        {showResult && (
          <div className="result-bar">
            <div className="result-stat-chips">
              <span className="stat-chip" title="Encoding speed">
                <Zap size={13} className="text-cyan" />
                {generationResult.performance?.fps || 0} FPS
              </span>
              <span className="stat-chip" title="Model inference time">
                <Gauge size={13} className="text-indigo" />
                {generationResult.performance?.inference_time || 0}s
              </span>
              <span className="stat-chip" title="File size">
                <Film size={13} />
                {generationResult.file_size_mb || 0} MB
              </span>
            </div>

            <button
              type="button"
              className="btn-icon"
              onClick={handleCopyLink}
              title="Copy download link"
              aria-label="Copy download link"
              disabled={!downloadUrl && !downloadBackupUrl}
            >
              <Link2 size={16} />
            </button>
            <button
              type="button"
              onClick={handleDownload}
              disabled={isDownloading || (!downloadUrl && !downloadBackupUrl)}
              className="btn-primary btn-sm"
            >
              <Download size={15} />
              <span>{isDownloading ? 'Downloading…' : 'Download MP4'}</span>
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
