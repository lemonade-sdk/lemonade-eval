/**
 * Tests for utility functions
 */

import { describe, it, expect } from 'vitest';
import {
  formatFileSize,
  formatDuration,
  formatPercent,
  formatCompactNumber,
  formatDate,
  formatDateTime,
  formatRelativeTime,
  truncateText,
  extractModelFamily,
  parseQuantization,
} from '../../utils';

describe('Utilities', () => {
  describe('formatFileSize', () => {
    it('should format bytes correctly', () => {
      expect(formatFileSize(0)).toBe('0 B');
      expect(formatFileSize(1024)).toBe('1 KB');
      expect(formatFileSize(1048576)).toBe('1 MB');
      expect(formatFileSize(1073741824)).toBe('1 GB');
    });
  });

  describe('formatDuration', () => {
    it('should format duration correctly', () => {
      expect(formatDuration(0)).toBe('< 1s');
      expect(formatDuration(30)).toBe('30s');
      expect(formatDuration(90)).toBe('1m 30s');
      expect(formatDuration(3661)).toBe('1.0h');
    });
  });

  describe('formatPercent', () => {
    it('should format percentage correctly', () => {
      expect(formatPercent(0.9567, 2)).toBe('0.96%');
      expect(formatPercent(99.999, 1)).toBe('100.0%');
    });
  });

  describe('formatCompactNumber', () => {
    it('should format large numbers', () => {
      expect(formatCompactNumber(1000)).toBe('1K');
      expect(formatCompactNumber(1500000)).toBe('1.5M');
      expect(formatCompactNumber(2000000000)).toBe('2B');
    });
  });

  describe('formatDate', () => {
    it('should handle null/undefined', () => {
      expect(formatDate(null)).toBe('-');
      expect(formatDate(undefined)).toBe('-');
    });

    it('should format date correctly', () => {
      const date = '2024-03-15T10:30:00Z';
      expect(formatDate(date)).toContain('2024');
    });
  });

  describe('formatDateTime', () => {
    it('should handle null/undefined', () => {
      expect(formatDateTime(null)).toBe('-');
      expect(formatDateTime(undefined)).toBe('-');
    });
  });

  describe('formatRelativeTime', () => {
    it('should handle null/undefined', () => {
      expect(formatRelativeTime(null)).toBe('-');
      expect(formatRelativeTime(undefined)).toBe('-');
    });
  });

  describe('truncateText', () => {
    it('should truncate long text', () => {
      expect(truncateText('Hello World', 5)).toBe('He...');
      expect(truncateText('Hi', 10)).toBe('Hi');
    });
  });

  describe('extractModelFamily', () => {
    it('should extract family from checkpoint', () => {
      expect(extractModelFamily('Llama-3.2-1B')).toBe('Llama');
      expect(extractModelFamily('Qwen2.5-7B')).toBe('Qwen');
      expect(extractModelFamily('Phi-3-mini')).toBe('Phi');
      expect(extractModelFamily('unknown-model')).toBe('Other');
    });
  });

  describe('parseQuantization', () => {
    it('should parse quantization from checkpoint', () => {
      expect(parseQuantization('model-int4.gguf')).toBe('int4');
      expect(parseQuantization('model-fp16.gguf')).toBe('fp16');
      expect(parseQuantization('model-awq.gguf')).toBe('awq');
      expect(parseQuantization('model-unknown')).toBeNull();
    });
  });
});
