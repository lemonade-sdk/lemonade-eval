/**
 * Utility functions for formatting and data transformation
 */

import { type ClassValue, clsx } from 'clsx';

/**
 * Combine class names conditionally
 */
export function cn(...inputs: ClassValue[]): string {
  return clsx(inputs);
}

/**
 * Format file size to human-readable format
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

/**
 * Format duration to human-readable format
 */
export function formatDuration(seconds: number): string {
  if (seconds < 1) return '< 1s';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  const hours = seconds / 3600;
  return `${hours.toFixed(1)}h`;
}

/**
 * Format percentage with fixed decimals
 */
export function formatPercent(value: number, decimals = 2): string {
  return `${value.toFixed(decimals)}%`;
}

/**
 * Format large numbers with K/M/B suffixes
 */
export function formatCompactNumber(num: number): string {
  return Intl.NumberFormat('en-US', {
    notation: 'compact',
    maximumFractionDigits: 2,
  }).format(num);
}

/**
 * Format date to readable string
 */
export function formatDate(date: string | Date | null | undefined, options?: Intl.DateTimeFormatOptions): string {
  if (!date) return '-';
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    ...options,
  });
}

/**
 * Format date with time
 */
export function formatDateTime(date: string | Date | null | undefined): string {
  if (!date) return '-';
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Format relative time (e.g., "2 hours ago")
 */
export function formatRelativeTime(date: string | Date | null | undefined): string {
  if (!date) return '-';
  const d = typeof date === 'string' ? new Date(date) : date;
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffSecs < 60) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return formatDate(date);
}

/**
 * Get status color for Mantine components
 */
export function getStatusColor(status: string): string {
  const colors: Record<string, string> = {
    // Run statuses
    pending: 'gray',
    running: 'blue',
    completed: 'green',
    failed: 'red',
    cancelled: 'orange',
    // Metric categories
    performance: 'blue',
    accuracy: 'green',
    efficiency: 'violet',
  };
  return colors[status.toLowerCase()] || 'gray';
}

/**
 * Get status badge variant
 */
export function getStatusVariant(status: string): string {
  const variants: Record<string, string> = {
    pending: 'outline',
    running: 'light',
    completed: 'light',
    failed: 'light',
    cancelled: 'outline',
  };
  return variants[status.toLowerCase()] || 'outline';
}

/**
 * Truncate text with ellipsis
 */
export function truncateText(text: string, maxLength: number): string {
  if (!text || text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 3)}...`;
}

/**
 * Generate a random ID
 */
export function generateId(): string {
  return Math.random().toString(36).substring(2, 11);
}

/**
 * Debounce function
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function debounce<T extends (...args: any[]) => any>(func: T, wait: number): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

/**
 * Check if value is a valid number
 */
export function isValidNumber(value: unknown): value is number {
  return typeof value === 'number' && !Number.isNaN(value) && Number.isFinite(value);
}

/**
 * Calculate percentage change
 */
export function calculatePercentChange(oldValue: number, newValue: number): number {
  if (oldValue === 0) return newValue > 0 ? 100 : 0;
  return ((newValue - oldValue) / oldValue) * 100;
}

/**
 * Get model family from checkpoint name
 */
export function extractModelFamily(checkpoint: string): string {
  const patterns: [RegExp, string][] = [
    [/llama/i, 'Llama'],
    [/qwen/i, 'Qwen'],
    [/phi/i, 'Phi'],
    [/mistral/i, 'Mistral'],
    [/gemma/i, 'Gemma'],
    [/falcon/i, 'Falcon'],
    [/mixtral/i, 'Mixtral'],
    [/yi-/i, 'Yi'],
  ];

  for (const [pattern, family] of patterns) {
    if (pattern.test(checkpoint)) return family;
  }

  return 'Other';
}

/**
 * Parse checkpoint to extract quantization info
 */
export function parseQuantization(checkpoint: string): string | null {
  const patterns: [RegExp, string][] = [
    [/int4/i, 'int4'],
    [/int8/i, 'int8'],
    [/fp16/i, 'fp16'],
    [/f16/i, 'fp16'],
    [/fp32/i, 'fp32'],
    [/f32/i, 'fp32'],
    [/awq/i, 'awq'],
    [/gptq/i, 'gptq'],
    [/gguf/i, 'gguf'],
  ];

  for (const [pattern, quant] of patterns) {
    if (pattern.test(checkpoint)) return quant;
  }

  return null;
}

/**
 * Group array by key
 */
export function groupBy<T, K extends keyof T>(array: T[], key: K): Map<T[K], T[]> {
  return array.reduce((map, item) => {
    const groupKey = item[key];
    const existing = map.get(groupKey) || [];
    map.set(groupKey, [...existing, item]);
    return map;
  }, new Map<T[K], T[]>());
}

/**
 * Sort array by key
 */
export function sortBy<T>(array: T[], key: keyof T, ascending = true): T[] {
  return [...array].sort((a, b) => {
    const aVal = a[key];
    const bVal = b[key];
    if (aVal < bVal) return ascending ? -1 : 1;
    if (aVal > bVal) return ascending ? 1 : -1;
    return 0;
  });
}
