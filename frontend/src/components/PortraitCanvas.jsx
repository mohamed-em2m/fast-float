import React, { useRef, useEffect, useState } from 'react';
import { Upload, Image as ImageIcon, X, Crop, Loader2 } from 'lucide-react';
import {
  blobToBase64,
  formatBytes,
  imageFormatFromFile,
} from '../services/api';

const SAMPLE_PORTRAITS = [
  { id: 'sam', name: 'Sam Altman', url: '/samples/portrait1.jpg' },
  { id: 'avatar', name: 'Studio Avatar', url: '/samples/portrait2.jpg' },
];

// Keep the JSON payload sane: base64 is ~33% larger than the file, and the
// backend decodes the whole thing in memory.
const MAX_IMAGE_BYTES = 15 * 1024 * 1024;

export default function PortraitCanvas({
  imagePreview,
  setImagePreview,
  setImageBase64,
  setImageFormat,
  noCrop,
  setNoCrop,
  step = 1,
  onToast,
}) {
  const fileInputRef = useRef(null);
  // Tracks blob: preview URLs we created so we can revoke them (static
  // /samples/* URLs must NOT be revoked).
  const blobUrlRef = useRef(null);
  // The latest selection wins: a slow FileReader must not overwrite the photo
  // the user picked afterwards.
  const selectionRef = useRef(0);
  const dragDepthRef = useRef(0);
  const [isDragging, setIsDragging] = useState(false);
  // null | 'file' | sample id
  const [loading, setLoading] = useState(null);

  useEffect(() => {
    return () => {
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
    };
  }, []);

  const setPreview = (url, { isBlob = false } = {}) => {
    if (blobUrlRef.current && blobUrlRef.current !== url) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }
    if (isBlob) blobUrlRef.current = url;
    setImagePreview(url);
  };

  const validate = (file) => {
    const format = imageFormatFromFile(file);
    if (!format) {
      return 'That file is not a supported image. Use a JPG, PNG or WebP photo.';
    }
    if (!file.size) {
      return 'That image is empty. Choose another photo.';
    }
    if (file.size > MAX_IMAGE_BYTES) {
      return `That image is ${formatBytes(file.size)} — the limit is ${formatBytes(MAX_IMAGE_BYTES)}.`;
    }
    return null;
  };

  const handleFileUpload = async (file) => {
    if (!file) return;
    const problem = validate(file);
    if (problem) {
      onToast?.(problem, 'error');
      return;
    }

    const token = ++selectionRef.current;
    setLoading('file');
    try {
      // Read first, then swap the preview: if the read fails the previous
      // portrait (and its base64) stays intact instead of a broken image.
      const base64 = await blobToBase64(file);
      if (token !== selectionRef.current) return;
      setPreview(URL.createObjectURL(file), { isBlob: true });
      setImageBase64(base64);
      setImageFormat?.(imageFormatFromFile(file) || 'jpg');
    } catch (err) {
      console.error('Failed to convert image to base64', err);
      if (token === selectionRef.current) {
        onToast?.('Could not read that image. Try another JPG, PNG or WebP file.', 'error');
      }
    } finally {
      if (token === selectionRef.current) setLoading(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    dragDepthRef.current = 0;
    setIsDragging(false);
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  };

  const handleSelectSample = async (sample) => {
    const token = ++selectionRef.current;
    setLoading(sample.id);
    try {
      const response = await fetch(sample.url, { cache: 'no-cache' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const blob = await response.blob();
      if (!blob.size) throw new Error('empty sample');
      const base64 = await blobToBase64(blob);
      if (token !== selectionRef.current) return;
      setPreview(sample.url);
      setImageBase64(base64);
      setImageFormat?.(imageFormatFromFile(blob) || 'jpg');
    } catch (err) {
      console.error('Failed to load sample image', err);
      if (token === selectionRef.current) {
        onToast?.(`Could not load "${sample.name}". Check that the sample images are available.`, 'error');
      }
    } finally {
      if (token === selectionRef.current) setLoading(null);
    }
  };

  const handleClearImage = () => {
    selectionRef.current += 1;
    setLoading(null);
    setPreview(null);
    setImageBase64(null);
    setImageFormat?.('jpg');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const busy = Boolean(loading);

  return (
    <div className="card-section glass-panel">
      <div className="section-header">
        <div className="section-title-wrap">
          <span className="step-index">{step}</span>
          <ImageIcon size={18} className="section-icon text-indigo" />
          <h2 className="section-title">Portrait</h2>
        </div>
        {imagePreview && (
          <button
            className="btn-icon-subtle"
            onClick={handleClearImage}
            title="Clear image"
            aria-label="Clear portrait"
            disabled={busy}
          >
            <X size={16} />
          </button>
        )}
      </div>

      {/* Drag & Drop Canvas */}
      <div
        className={`portrait-dropzone ${imagePreview ? 'has-image' : ''} ${isDragging ? 'is-dragging' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragEnter={() => {
          dragDepthRef.current += 1;
          setIsDragging(true);
        }}
        onDragLeave={() => {
          dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
          if (!dragDepthRef.current) setIsDragging(false);
        }}
        onDrop={handleDrop}
        onClick={() => !imagePreview && !busy && fileInputRef.current?.click()}
        aria-busy={busy}
      >
        <input
          type="file"
          ref={fileInputRef}
          accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp"
          onChange={(e) => {
            const file = e.target.files?.[0];
            // Reset first so picking the same file twice still fires onChange.
            e.target.value = '';
            handleFileUpload(file);
          }}
          style={{ display: 'none' }}
        />

        {imagePreview ? (
          <div className="portrait-preview-container">
            <img src={imagePreview} alt="Portrait Reference" className="portrait-preview-img" />
            <div className="portrait-overlay">
              <div className="portrait-focus-ring" />
              <button
                type="button"
                className="btn-change-image"
                onClick={(e) => {
                  e.stopPropagation();
                  if (!busy) fileInputRef.current?.click();
                }}
                disabled={busy}
              >
                Change Photo
              </button>
            </div>
          </div>
        ) : (
          <div className="portrait-placeholder">
            <div className="upload-icon-circle">
              <Upload size={24} />
            </div>
            <p className="upload-hint-main">Drop portrait image here, or <span>browse</span></p>
            <p className="upload-hint-sub">PNG, JPG, WebP (Frontal face works best)</p>
          </div>
        )}

        {busy && (
          <div className="portrait-loading" role="status" aria-live="polite">
            <Loader2 size={20} className="spin" />
            <span>Reading image…</span>
          </div>
        )}
      </div>

      {/* Face Crop Toggle */}
      <div className="crop-toggle-row">
        <label className="crop-toggle-label">
          <input
            type="checkbox"
            checked={!noCrop}
            onChange={(e) => setNoCrop(!e.target.checked)}
            className="crop-checkbox"
          />
          <Crop size={15} />
          <span>Auto Face Crop & Centering</span>
        </label>
        <span className="crop-info-hint">Recommended for non-aligned photos</span>
      </div>

      {/* Sample Presets */}
      <div className="samples-row">
        <span className="samples-label">Quick Samples:</span>
        <div className="samples-pill-list">
          {SAMPLE_PORTRAITS.map((sample) => (
            <button
              key={sample.id}
              type="button"
              className={`sample-pill ${imagePreview === sample.url ? 'active' : ''}`}
              onClick={() => handleSelectSample(sample)}
              disabled={busy}
            >
              {loading === sample.id ? (
                <Loader2 size={14} className="spin" />
              ) : (
                <img src={sample.url} alt="" className="sample-thumbnail" />
              )}
              <span>{sample.name}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
