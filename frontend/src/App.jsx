import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import PortraitCanvas from './components/PortraitCanvas';
import AudioEngine from './components/AudioEngine';
import ParameterControls from './components/ParameterControls';
import StudioStage from './components/StudioStage';
import TelemetryDashboard from './components/TelemetryDashboard';
import { generateVideo, blobToBase64, alignUrlToApi } from './services/api';

const DEFAULT_PORTRAIT = '/samples/portrait1.jpg';

export default function App() {
  // Input State
  const [imagePreview, setImagePreview] = useState(DEFAULT_PORTRAIT);
  const [imageBase64, setImageBase64] = useState(null);
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
  });

  // Generation & Output State
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationResult, setGenerationResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [toast, setToast] = useState(null);

  const showToast = (message, type = 'info') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 4000);
  };

  // Preload the default sample portrait so the preview the user SEES is the
  // image the backend RECEIVES. Previously imageBase64 stayed null and the
  // server silently fell back to its placeholder image.
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
      showToast('Please record microphone audio or upload a speech clip first.', 'error');
      return;
    }

    setIsGenerating(true);
    try {
      const payload = {
        audio_base64: audioBase64,
        audio_format: audioFormat,
        sample_rate: 16000,
        channels: 1,
        sample_width: 2,
        image_base64: imageBase64 || null,
        image_format: 'jpg',
        emotion: params.emotion,
        nfe: params.nfe,
        seed: params.seed,
        a_cfg_scale: params.a_cfg_scale,
        r_cfg_scale: params.r_cfg_scale,
        e_cfg_scale: params.e_cfg_scale,
        no_crop: noCrop,
        compress: params.compress,
        quality: params.quality,
      };

      const result = await generateVideo(payload);
      // Point playback/download URLs at the API host the browser can reach.
      result.stream_url = alignUrlToApi(result.stream_url);
      result.download_url = alignUrlToApi(result.download_url);
      if (result.download_backup) result.download_backup = alignUrlToApi(result.download_backup);
      setGenerationResult(result);
      setHistory((prev) => [result, ...prev]);
      showToast(`Talking portrait generated successfully in ${result.performance?.total_time}s!`, 'success');
    } catch (err) {
      console.error('Generation failed', err);
      showToast(err.message || 'Generation failed. Check server status.', 'error');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="app-container">
      {/* Toast Notification */}
      {toast && (
        <div className={`toast-notification toast-${toast.type}`}>
          <span>{toast.message}</span>
        </div>
      )}

      {/* Navigation Header */}
      <Header onToast={showToast} />

      {/* Main Studio Grid */}
      <main className="studio-main-grid">
        {/* Left Column: Creator Workbench */}
        <section className="workbench-col">
          <PortraitCanvas 
            imagePreview={imagePreview}
            setImagePreview={setImagePreview}
            setImageBase64={setImageBase64}
            noCrop={noCrop}
            setNoCrop={setNoCrop}
          />

          <AudioEngine
            audioBlob={audioBlob}
            setAudioBlob={setAudioBlob}
            audioBase64={audioBase64}
            setAudioBase64={setAudioBase64}
            setAudioFormat={setAudioFormat}
          />

          <ParameterControls 
            params={params}
            setParams={setParams}
          />
        </section>

        {/* Right Column: Stage & Telemetry */}
        <section className="stage-col">
          <StudioStage 
            onGenerate={handleGenerate}
            isGenerating={isGenerating}
            generationResult={generationResult}
            hasAudio={Boolean(audioBase64)}
            onToast={showToast}
          />

          <TelemetryDashboard 
            latestResult={generationResult}
            history={history}
          />
        </section>
      </main>
    </div>
  );
}
