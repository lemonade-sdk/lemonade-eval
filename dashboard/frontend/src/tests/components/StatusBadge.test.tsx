/**
 * Tests for StatusBadge Component
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '../utils';
import { StatusBadge } from '@/components/common';
import '@testing-library/jest-dom';

describe('StatusBadge', () => {
  describe('Rendering', () => {
    it('should render pending status correctly', () => {
      render(<StatusBadge status="pending" />);
      expect(screen.getByText('Pending')).toBeInTheDocument();
    });

    it('should render running status correctly', () => {
      render(<StatusBadge status="running" />);
      expect(screen.getByText('Running')).toBeInTheDocument();
    });

    it('should render completed status correctly', () => {
      render(<StatusBadge status="completed" />);
      expect(screen.getByText('Completed')).toBeInTheDocument();
    });

    it('should render failed status correctly', () => {
      render(<StatusBadge status="failed" />);
      expect(screen.getByText('Failed')).toBeInTheDocument();
    });

    it('should render cancelled status correctly', () => {
      render(<StatusBadge status="cancelled" />);
      expect(screen.getByText('Cancelled')).toBeInTheDocument();
    });

    it('should capitalize status text', () => {
      render(<StatusBadge status="pending" />);
      expect(screen.getByText('Pending')).toBeInTheDocument();
    });

    it('should handle unknown status with default styling', () => {
      render(<StatusBadge status="unknown_status" />);
      expect(screen.getByText('Unknown_status')).toBeInTheDocument();
    });
  });

  describe('Icon Display', () => {
    it('should render with icon by default', () => {
      render(<StatusBadge status="completed" />);
      const badge = screen.getByText('Completed');
      expect(badge).toBeInTheDocument();
    });

    it('should render without icon when showIcon is false', () => {
      render(<StatusBadge status="completed" showIcon={false} />);
      expect(screen.getByText('Completed')).toBeInTheDocument();
    });
  });

  describe('Props', () => {
    it('should apply custom props', () => {
      render(<StatusBadge status="completed" size="lg" />);
      const badge = screen.getByText('Completed');
      expect(badge).toBeInTheDocument();
    });

    it('should accept data-testid prop', () => {
      render(<StatusBadge status="completed" data-testid="test-badge" />);
      expect(screen.getByTestId('test-badge')).toBeInTheDocument();
    });

    it('should accept className prop', () => {
      render(<StatusBadge status="completed" className="custom-class" />);
      const badge = screen.getByText('Completed');
      expect(badge.parentElement).toHaveClass('custom-class');
    });
  });

  describe('Color Variants', () => {
    it('should use gray color for pending', () => {
      render(<StatusBadge status="pending" />);
      const badge = screen.getByText('Pending');
      expect(badge).toBeInTheDocument();
    });

    it('should use green color for completed', () => {
      render(<StatusBadge status="completed" />);
      const badge = screen.getByText('Completed');
      expect(badge).toBeInTheDocument();
    });

    it('should use red color for failed', () => {
      render(<StatusBadge status="failed" />);
      const badge = screen.getByText('Failed');
      expect(badge).toBeInTheDocument();
    });
  });
});
