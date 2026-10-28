import { describe, expect, it } from 'vitest';

import {
  DEFAULT_PORTRAIT_URL,
  SAMPLE_PORTRAITS,
  labelFromId,
} from './samplePortraits';

describe('sample portrait discovery', () => {
  it('finds at least one portrait in src/assets/samples', () => {
    expect(SAMPLE_PORTRAITS.length).toBeGreaterThan(0);
  });

  it('gives every portrait a unique id, a label and a bundled url', () => {
    const ids = SAMPLE_PORTRAITS.map((sample) => sample.id);
    expect(new Set(ids).size).toBe(ids.length);

    for (const sample of SAMPLE_PORTRAITS) {
      // The id is the file stem, so no extension survives into keys or labels.
      expect(sample.id).not.toMatch(/\./);
      expect(sample.name.length).toBeGreaterThan(0);
      expect(typeof sample.url).toBe('string');
      expect(sample.url.length).toBeGreaterThan(0);
    }
  });

  it('sorts numerically so portrait10 comes after portrait9', () => {
    const ids = SAMPLE_PORTRAITS.map((sample) => sample.id);
    const sorted = [...ids].sort((a, b) => a.localeCompare(b, 'en', { numeric: true }));
    expect(ids).toEqual(sorted);
  });

  it('uses the first portrait as the default', () => {
    expect(DEFAULT_PORTRAIT_URL).toBe(SAMPLE_PORTRAITS[0].url);
  });
});

describe('labelFromId', () => {
  it('turns a file stem into a readable label', () => {
    expect(labelFromId('portrait1')).toBe('Portrait 1');
    expect(labelFromId('portrait-10')).toBe('Portrait 10');
    expect(labelFromId('sam_altman')).toBe('Sam Altman');
  });

  it('never returns an empty label', () => {
    expect(labelFromId('')).toBe('Sample');
  });
});
