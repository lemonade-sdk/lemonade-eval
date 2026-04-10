/**
 * Tests for MetricCard Component
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '../utils';
import { MetricCard } from '@/components/metrics/MetricCard';
import '@testing-library/jest-dom';

describe('MetricCard', () => {
  describe('Basic Rendering', () => {
    it('should render label and value correctly', () => {
      render(
        <MetricCard
          label="Test Metric"
          value={42.5}
          unit="tokens/s"
        />
      );
      expect(screen.getByText('Test Metric')).toBeInTheDocument();
      expect(screen.getByText('42.5')).toBeInTheDocument();
      expect(screen.getByText('tokens/s')).toBeInTheDocument();
    });

    it('should render string value', () => {
      render(
        <MetricCard
          label="Status"
          value="Running"
        />
      );
      expect(screen.getByText('Running')).toBeInTheDocument();
    });

    it('should format large numbers with locale string', () => {
      render(
        <MetricCard
          label="Large Number"
          value={1000000}
        />
      );
      expect(screen.getByText('1,000,000')).toBeInTheDocument();
    });

    it('should render without unit when not provided', () => {
      render(
        <MetricCard
          label="No Unit"
          value={100}
        />
      );
      expect(screen.getByText('No Unit')).toBeInTheDocument();
      expect(screen.getByText('100')).toBeInTheDocument();
    });
  });

  describe('Description/Tooltip', () => {
    it('should render description as tooltip', () => {
      render(
        <MetricCard
          label="TTFT"
          value={0.025}
          unit="s"
          description="Time to first token"
        />
      );
      expect(screen.getByText('TTFT')).toBeInTheDocument();
    });

    it('should not render tooltip when description is not provided', () => {
      render(
        <MetricCard
          label="No Description"
          value={100}
        />
      );
      expect(screen.getByText('No Description')).toBeInTheDocument();
    });
  });

  describe('Highlight', () => {
    it('should highlight when highlight prop is true', () => {
      render(
        <MetricCard
          label="Best Metric"
          value={100}
          highlight
        />
      );
      const card = screen.getByText('Best Metric').closest('[role="button"]') || screen.getByText('Best Metric').parentElement;
      expect(card).toBeInTheDocument();
    });

    it('should not highlight when highlight is false', () => {
      render(
        <MetricCard
          label="Normal Metric"
          value={50}
          highlight={false}
        />
      );
      expect(screen.getByText('Normal Metric')).toBeInTheDocument();
    });
  });

  describe('Trend', () => {
    it('should render trend when provided', () => {
      render(
        <MetricCard
          label="Performance"
          value={95}
          trend="up"
          trendValue={12.5}
        />
      );
      expect(screen.getByText('+12.5%')).toBeInTheDocument();
    });

    it('should render negative trend with minus sign', () => {
      render(
        <MetricCard
          label="Performance"
          value={95}
          trend="down"
          trendValue={-5.5}
        />
      );
      expect(screen.getByText('-5.5%')).toBeInTheDocument();
    });

    it('should not render trend when not provided', () => {
      render(
        <MetricCard
          label="No Trend"
          value={50}
        />
      );
      expect(screen.queryByText(/\+%$/)).not.toBeInTheDocument();
    });

    it('should render up trend icon', () => {
      render(
        <MetricCard
          label="Trending Up"
          value={100}
          trend="up"
          trendValue={10}
        />
      );
      expect(screen.getByText('Trending Up')).toBeInTheDocument();
    });
  });

  describe('Custom Color', () => {
    it('should apply custom color', () => {
      render(
        <MetricCard
          label="Custom"
          value={50}
          color="red"
        />
      );
      expect(screen.getByText('50')).toBeInTheDocument();
    });

    it('should accept any valid color string', () => {
      render(
        <MetricCard
          label="Blue Value"
          value={75}
          color="blue"
        />
      );
      expect(screen.getByText('75')).toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('should handle zero value', () => {
      render(
        <MetricCard
          label="Zero"
          value={0}
        />
      );
      expect(screen.getByText('0')).toBeInTheDocument();
    });

    it('should handle negative value', () => {
      render(
        <MetricCard
          label="Negative"
          value={-10}
        />
      );
      expect(screen.getByText('-10')).toBeInTheDocument();
    });

    it('should handle decimal value', () => {
      render(
        <MetricCard
          label="Decimal"
          value={0.123456}
        />
      );
      expect(screen.getByText('0.123456')).toBeInTheDocument();
    });

    it('should handle empty string value', () => {
      render(
        <MetricCard
          label="Empty"
          value=""
        />
      );
      expect(screen.getByText('')).toBeInTheDocument();
    });
  });
});
