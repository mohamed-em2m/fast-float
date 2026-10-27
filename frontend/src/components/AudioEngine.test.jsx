import React, { StrictMode } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import AudioEngine from './AudioEngine';

function renderEngine(overrides = {}) {
  const props = {
    audioBlob: null,
    setAudioBlob: vi.fn(),
    audioBase64: null,
    setAudioBase64: vi.fn(),
    setAudioFormat: vi.fn(),
    onToast: vi.fn(),
    ...overrides,
  };
  // main.jsx wraps the app in StrictMode; the double mount/cleanup must not
  // make the engine believe it was unmounted.
  render(
    <StrictMode>
      <AudioEngine {...props} />
    </StrictMode>
  );
  return props;
}

function pickFile(name, type, bytes = [1, 2, 3, 4]) {
  const input = document.querySelector('input[type="file"]');
  const file = new File([new Uint8Array(bytes)], name, { type });
  fireEvent.change(input, { target: { files: [file] } });
}

describe('AudioEngine', () => {
  it('publishes an uploaded clip so generation can start (StrictMode)', async () => {
    const props = renderEngine();
    fireEvent.click(screen.getByRole('button', { name: /upload file/i }));

    pickFile('voice.wav', 'audio/wav');

    await waitFor(() => expect(props.setAudioBase64).toHaveBeenCalled());
    expect(props.setAudioBase64).toHaveBeenCalledWith('AQIDBA==');
    expect(props.setAudioFormat).toHaveBeenCalledWith('wav');
    expect(props.setAudioBlob).toHaveBeenCalled();
  });

  it('publishes the sample clip so generation can start (StrictMode)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        blob: async () => new Blob([new Uint8Array([1, 2, 3])], { type: 'audio/wav' }),
      }))
    );
    const props = renderEngine();

    fireEvent.click(screen.getByRole('button', { name: /sample speech/i }));
    fireEvent.click(screen.getByRole('button', { name: /use sample/i }));

    await waitFor(() => expect(props.setAudioBase64).toHaveBeenCalled());
    expect(props.setAudioBase64).toHaveBeenCalledWith('AQID');
    expect(props.setAudioFormat).toHaveBeenCalledWith('wav');
    expect(props.onToast).toHaveBeenCalledWith(expect.stringMatching(/sample clip loaded/i), 'success');
  });

  it('rejects an unsupported file without touching the selection', async () => {
    const props = renderEngine();
    fireEvent.click(screen.getByRole('button', { name: /upload file/i }));

    pickFile('notes.txt', 'text/plain');

    await waitFor(() =>
      expect(props.onToast).toHaveBeenCalledWith(expect.stringMatching(/unsupported audio file/i), 'error')
    );
    expect(props.setAudioBase64).not.toHaveBeenCalled();
  });
});
