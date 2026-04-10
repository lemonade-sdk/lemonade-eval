/**
 * Tests for LoadingSpinner Component
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '../utils';
import { LoadingSpinner } from '@/components/common';
import '@testing-library/jest-dom';

describe('LoadingSpinner', () => {
  it('should render spinner without message', () => {
    render(<LoadingSpinner />);
    const loader = screen.getByRole('progressbar');
    expect(loader).toBeInTheDocument();
  });

  it('should render spinner with message', () => {
    render(<LoadingSpinner message="Loading data..." />);
    expect(screen.getByText('Loading data...')).toBeInTheDocument();
  });

  it('should render in fullScreen mode', () => {
    render(<LoadingSpinner fullScreen />);
    // Full screen should take viewport dimensions
    const container = screen.getByRole('progressbar').parentElement;
    expect(container).toBeInTheDocument();
  });

  it('should render with different sizes', () => {
    const { rerender } = render(<LoadingSpinner size="sm" />);
    expect(screen.getByRole('progressbar')).toBeInTheDocument();

    rerender(<LoadingSpinner size="lg" />);
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });
});
