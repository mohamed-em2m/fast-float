import React, { useState, useEffect, useRef } from 'react';
import { CheckCircle2, Info, TriangleAlert, X } from 'lucide-react';
import Header from './components/Header';
import PortraitCanvas from './components/PortraitCanvas';
import AudioEngine from './components/AudioEngine';
import ParameterControls from './components/ParameterControls';
import StudioStage from './components/StudioStage';
import InsightsPanel from './components/InsightsPanel';
import LiveAudioStudio from './components/LiveAudioStudio';
import { generateVideo, blobToBase64, alignUrlToApi, fetchProgress } from './services/api';

const DEFAULT_PORTRAIT = '/samples/portrait1.jpg';
const PROGRESS_POLL_MS = 1000;
const MODE_KEY = 'float_mode';

const TOAST_ICON = {
  success: CheckCircle2,
  error: TriangleAlert,
  info: Info,
};

export default function App() {
  // Workspace mode: async studio render vs. live microphone avatar.
  const [mode, setMode] = useState(() => {
    try {
      return localStorage.getItem(MODE_KEY) === 'live' ? 'live' : 'studio';
    } catch {
      return 'studio';
    }
  });

  // Input State
  const [imagePreview, setImagePreview] = useState(DEFAULT_PORTRAIT);
  const [imageBase64, setImageBase64] = useState(null);
  // Container hint for image_base64; portrait uploads set the real format.
  const [imageFormat, setImageFormat] = useState('jpg');
  const [noCrop, setNoCrop] = useState(false);

  const [audioBlob, setAudioBlob] = useState(null);
  const [audioBase64, setAudioBase64] = useState(null);
  const [audioFormat, setAudioFormat] = useState('wav');

  // Parameters
  const [params, setParams] = useState({
    emotion: 'S2E',
    nfe: 10,
    seed: 25,
    a_cfg_scale: 2.0,
    r_cfg_scale: 1.0,
    e_cfg_scale: 1.0,
    compress: true,
    quality: 'balanced',
    chunk_seconds: 0,
    decode_batch_size: 0,
    chunk_batch_size: 1,
  });

  // Generation & Output State
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(null);
  const [generationResult, setGenerationResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [toast, setToast] = useState(null);
  // 'auto' streams the finished file; live transports stream frames live.
  const [transport, setTransport] = useState('auto');
  const [liveRequestId, setLiveRequestId] = useState(null);
  const progressTimerRef = useRef(null);
  const toastTimerRef = useRef(null);

  const stopProgressPolling = () => {
    if (progressTimerRef.current) {
      clearInterval(progressTimerRef.current);
      progressTimerRef.current = null;
    }
  };

  // Clear any in-flight progress polling on unmount.
  useEffect(() => stopProgressPolling, []);

  useEffect(() => {
    try {
      localStorage.setItem(MODE_KEY, mode);
    } catch {
      /* private mode */
    }
  }, [mode]);

  const showToast = (message, type = 'info') => {
    if (!message) return;
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    setToast({ message, type });
    toastTimerRef.current = setTimeout(() => setToast(null), 4500);
  };

  useEffect(() => () => {
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
  }, []);

  const handleModeChange = (next) => {
    if (next === mode) return;
    if (isGenerating) {
      showToast('Let the current render finish before switching modes.', 'info');
      return;
    }
    setMode(next);
  };

  // Preload the default sample portrait so the preview the user SEES is the
  // image the backend RECEIVES.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(DEFAULT_PORTRAIT);
        if (!res.ok) return;
        const blob = await res.blob();
        const base64 = await blobToBase64(blob);
        if (!cancelled) setImageBase64((prev) => prev ?? base64);
      } catch (e) {
        console.error('Failed to preload default portrait', e);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const handleGenerate = async () => {
    if (!audioBase64) {
      showToast('Add a voice clip first — record, upload or use the sample.', 'error');
      return;
    }

    setIsGenerating(true);
    // A client id lets us poll /progress while the long POST is still open.
    const clientRequestId =
      (typeof crypto !== 'undefined' && crypto.randomUUID)
        ? crypto.randomUUID()
        : `c${Date.now()}${Math.random().toString(16).slice(2)}`;
    setProgress({ status: 'queued', stage: 'queued', percent: 0, completed_chunks: 0, total_chunks: 0 });
    const live = transport !== 'mp4';
    setLiveRequestId(live ? clientRequestId : null);
    stopProgressPolling();
    progressTimerRef.current = setInterval(async () => {
      const p = await fetchProgress(clientRequestId);
      if (p) setProgress(p);
    }, PROGRESS_POLL_MS);

    try {
      const payload = {
        audio_base64: audioBase64,
        audio_format: audioFormat,
        sample_rate: 16000,
        channels: 1,
        sample_width: 2,
        image_base64: imageBase64 || null,
        image_format: imageFormat || 'jpg',
        emotion: params.emotion,
        nfe: params.nfe,
        seed: params.seed,
        a_cfg_scale: params.a_cfg_scale,
        r_cfg_scale: params.r_cfg_scale,
        e_cfg_scale: params.e_cfg_scale,
        no_crop: noCrop,
        compress: params.compress,
        quality: params.quality,
        chunk_seconds: params.chunk_seconds,
        chunk_overlap_seconds: 0.5,
        decode_batch_size: params.decode_batch_size,
        chunk_batch_size: params.chunk_batch_size,
        client_request_id: clientRequestId,
        live,
      };

      const result = await generateVideo(payload);
      result.stream_url = alignUrlToApi(result.stream_url);
      result.download_url = alignUrlToApi(result.download_url);
      if (result.download_backup) result.download_backup = alignUrlToApi(result.download_backup);
      setGenerationResult(result);
      setHistory((prev) => [result, ...prev]);
      const chunkNote = result.chunks > 1 ? ` · ${result.chunks} chunks` : '';
      showToast(`Talking portrait ready in ${result.performance?.total_time ?? '—'}s${chunkNote}.`, 'success');
    } catch (err) {
      console.error('Generation failed', err);
      showToast(err.message || 'Generation failed. Check that the server is running.', 'error');
      // A failed live request must not leave a dead preview attached.
      if (live) setLiveRequestId(null);
    } finally {
      stopProgressPolling();
      setProgress(null);
      setIsGenerating(false);
      // Keep `liveRequestId` until the live transport reports 'done': a
      // real-time stream is still playing the frames buffered during render.
    }
  };

  const portraitCard = (
    <PortraitCanvas
      imagePreview={imagePreview}
      setImagePreview={setImagePreview}
      setImageBase64={setImageBase64}
      setImageFormat={setImageFormat}
      noCrop={noCrop}
      setNoCrop={setNoCrop}
      step={1}
      onToast={showToast}
    />
  );

  const ToastIcon = toast ? TOAST_ICON[toast.type] || Info : Info;

  return (
    <div className="app-shell">
      {toast && (
        <div
          className={`toast-notification toast-${toast.type}`}
          role="status"
          aria-live="polite"
        >
          <ToastIcon size={17} className="toast-icon" />
          <span>{toast.message}</span>
          <button
            type="button"
            className="toast-close"
            onClick={() => setToast(null)}
            aria-label="Dismiss notification"
          >
            <X size={14} />
          </button>
        </div>
      )}

      <Header onToast={showToast} mode={mode} onModeChange={handleModeChange} />

      {mode === 'live' ? (
        <LiveAudioStudio
          imagePreview={imagePreview}
          imageBase64={imageBase64}
          imageFormat={imageFormat}
          onToast={showToast}
          onOpenStudio={() => setMode('studio')}
          portraitPicker={portraitCard}
        />
      ) : (
        <main className="studio-workspace">
          <aside className="studio-composer">
            {portraitCard}

            <AudioEngine
              audioBlob={audioBlob}
              setAudioBlob={setAudioBlob}
              audioBase64={audioBase64}
              setAudioBase64={setAudioBase64}
              setAudioFormat={setAudioFormat}
              step={2}
              onToast={showToast}
            />

            <ParameterControls params={params} setParams={setParams} step={3} />
          </aside>

          <div className="studio-stage-column">
            <StudioStage
              onGenerate={handleGenerate}
              isGenerating={isGenerating}
              progress={progress}
              generationResult={generationResult}
              hasAudio={Boolean(audioBase64)}
              onToast={showToast}
              transport={transport}
              setTransport={setTransport}
              liveRequestId={liveRequestId}
              imagePreview={imagePreview}
            />

            <InsightsPanel
              latestResult={generationResult}
              history={history}
              onToast={showToast}
            />
          </div>
        </main>
      )}

      <footer className="app-footer">
        <span>
          Built on <strong>FLOAT</strong> (ICCV 2025) — research use only.
        </span>
        <a href="https://github.com/deepbrainai-research/float" target="_blank" rel="noreferrer">
          Original project &amp; credits
        </a>
      </footer>
    </div>
  );
}
