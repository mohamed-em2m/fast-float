import { beforeEach, describe, expect, it, vi } from 'vitest';

async function loadApi() {
  vi.resetModules();
  return import('./api');
}

beforeEach(() => {
  try {
    window.localStorage.clear();
  } catch {
    /* ignore */
  }
  window.history.replaceState({}, '', '/');
});

describe('API base resolution', () => {
  it('defaults to the same-origin /api proxy', async () => {
    const api = await loadApi();
    expect(api.getApiUrl()).toBe(`${window.location.origin}/api`);
    expect(api.isDefaultApiUrl()).toBe(true);
  });

  it('adopts ?api= and strips it from the browser URL', async () => {
    window.history.replaceState(
      {},
      '',
      `/?api=${encodeURIComponent('https://api.example.com/')}`
    );

    const api = await loadApi();
    expect(api.getApiUrl()).toBe('https://api.example.com');
    expect(window.location.search).toBe('');
    expect(window.localStorage.getItem('float_api_url')).toBe('https://api.example.com');
  });

  it('rejects a stored page URL instead of hijacking the API base', async () => {
    window.localStorage.setItem(
      'float_api_url',
      'http://localhost:3000/?api=https://evil.example.com'
    );

    const api = await loadApi();
    expect(api.getApiUrl()).toBe(`${window.location.origin}/api`);
    expect(window.localStorage.getItem('float_api_url')).toBeNull();
  });

  it('rewrites loopback backend URLs onto the adopted API base', async () => {
    window.history.replaceState(
      {},
      '',
      `/?api=${encodeURIComponent('https://api.example.com')}`
    );
    const api = await loadApi();

    expect(api.alignUrlToApi('http://localhost:8000/download/abc?x=1'))
      .toBe('https://api.example.com/download/abc?x=1');
    expect(api.alignUrlToApi('https://cdn.example.com/video.mp4'))
      .toBe('https://cdn.example.com/video.mp4');
    expect(api.alignUrlToApi('/relative/path')).toBe('https://api.example.com/relative/path');
  });

  it('offers loopback host variants for a stored loopback base', async () => {
    window.localStorage.setItem('float_api_url', 'http://localhost:9000');
    const api = await loadApi();

    expect(api.getApiCandidates()).toEqual([
      'http://localhost:9000',
      'http://127.0.0.1:9000',
      'http://0.0.0.0:9000',
    ]);
  });

  it('notifies listeners when the base changes', async () => {
    const api = await loadApi();
    const listener = vi.fn();
    window.addEventListener('float:api_url_changed', listener);

    api.setApiUrl('http://localhost:8123/');
    expect(api.getApiUrl()).toBe('http://localhost:8123');
    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener.mock.calls[0][0].detail).toBe('http://localhost:8123');

    window.removeEventListener('float:api_url_changed', listener);
  });
});

describe('file format helpers', () => {
  it('maps image containers and rejects unsupported ones', async () => {
    const api = await loadApi();
    expect(api.imageFormatFromFile({ type: 'image/png', name: 'a.png' })).toBe('png');
    expect(api.imageFormatFromFile({ type: '', name: 'photo.JPEG' })).toBe('jpg');
    expect(api.imageFormatFromFile({ type: 'image/gif', name: 'a.gif' })).toBeNull();
  });

  it('maps audio containers, including codec parameters', async () => {
    const api = await loadApi();
    expect(api.audioFormatFromFile({ type: 'audio/webm;codecs=opus', name: 'a' })).toBe('webm');
    expect(api.audioFormatFromFile({ type: '', name: 'voice.m4a' })).toBe('mp4');
    expect(api.audioFormatFromFile({ type: 'application/zip', name: 'a.zip' })).toBeNull();
  });

  it('formats byte sizes for validation messages', async () => {
    const api = await loadApi();
    expect(api.formatBytes(512)).toBe('512 B');
    expect(api.formatBytes(2048)).toBe('2 KB');
    expect(api.formatBytes(3 * 1024 * 1024)).toBe('3.0 MB');
  });
});
