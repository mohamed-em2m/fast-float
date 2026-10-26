import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  clearStoredMemory,
  getStored,
  removeStored,
  setStored,
} from './storage';

function restoreStorage(descriptor) {
  if (descriptor) {
    Object.defineProperty(window, 'localStorage', descriptor);
  } else {
    delete window.localStorage;
  }
}

describe('safe storage', () => {
  beforeEach(() => {
    clearStoredMemory();
    try {
      window.localStorage.clear();
    } catch {
      /* ignore */
    }
  });

  it('round-trips values through localStorage', () => {
    expect(setStored('float_test', 'hello')).toBe(true);
    expect(getStored('float_test')).toBe('hello');
    removeStored('float_test');
    expect(getStored('float_test', 'fallback')).toBe('fallback');
  });

  it('returns the fallback for a missing key', () => {
    expect(getStored('missing', 'fallback')).toBe('fallback');
    expect(getStored('missing')).toBeNull();
  });

  it('never throws when localStorage access itself throws', () => {
    const original = Object.getOwnPropertyDescriptor(window, 'localStorage');
    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      get() {
        throw new Error('SecurityError: storage is blocked');
      },
    });

    try {
      expect(() => getStored('float_mode')).not.toThrow();
      expect(getStored('float_mode', 'studio')).toBe('studio');

      // Writes fall back to the in-memory mirror, so the current tab still
      // behaves as if the setting was saved.
      expect(setStored('float_mode', 'live')).toBe(false);
      expect(getStored('float_mode')).toBe('live');
      removeStored('float_mode');
      expect(getStored('float_mode', 'studio')).toBe('studio');
    } finally {
      restoreStorage(original);
    }
  });

  it('keeps working when writes fail (quota / private mode)', () => {
    const setItem = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('QuotaExceededError');
    });

    try {
      expect(setStored('float_api_url', 'https://example.com')).toBe(false);
      // The session mirror wins over the stale on-disk value.
      expect(getStored('float_api_url')).toBe('https://example.com');
    } finally {
      setItem.mockRestore();
    }
  });
});
