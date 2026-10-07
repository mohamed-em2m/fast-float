import React, { useState } from 'react';
import Header from './components/Header';
import PortraitCanvas from './components/PortraitCanvas';
import AudioEngine from './components/AudioEngine';
import ParameterControls from './components/ParameterControls';
import StudioStage from './components/StudioStage';
import TelemetryDashboard from './components/TelemetryDashboard';
import { generateVideo } from './services/api';

export default function App() {
  // Input State
  const [imagePreview, setImagePreview] = useState('/samples/portrait1.jpg');
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
            audioFormat={audioFormat}
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
            hasImage={Boolean(imagePreview)}
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
