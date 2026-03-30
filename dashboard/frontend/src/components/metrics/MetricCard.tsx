/**
 * Metric Card Component
 * Displays a single metric value with label and optional trend
 */

import { Card, Group, Text, Box, Badge, Tooltip } from '@mantine/core';
import {
  IconTrendingUp,
  IconTrendingDown,
  IconMinus,
} from '@tabler/icons-react';

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  description?: string;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: number;
  highlight?: boolean;
  color?: string;
}

export function MetricCard({
  label,
  value,
  unit,
  description,
  trend,
  trendValue,
  highlight = false,
  color,
}: MetricCardProps) {
  const TrendIcon = trend === 'up'
    ? IconTrendingUp
    : trend === 'down'
      ? IconTrendingDown
      : IconMinus;

  const trendColor = trend === 'up'
    ? 'green'
    : trend === 'down'
      ? 'red'
      : 'gray';

  return (
    <Card
      padding="lg"
      radius="md"
      withBorder
      bg={highlight ? 'var(--mantine-color-blue-light)' : undefined}
      style={{
        borderColor: highlight ? 'var(--mantine-color-blue-6)' : undefined,
      }}
    >
      <Group justify="space-between" wrap="nowrap">
        <Box>
          <Tooltip label={description} disabled={!description}>
            <Text size="xs" c="dimmed" fw={500} tt="uppercase">
              {label}
            </Text>
          </Tooltip>
          <Group gap="xs" align="flex-end" wrap="nowrap">
            <Text size="xl" fw={700} c={color}>
              {typeof value === 'number' ? value.toLocaleString() : value}
            </Text>
            {unit && (
              <Text size="xs" c="dimmed" fw={500}>
                {unit}
              </Text>
            )}
          </Group>
        </Box>
        {trend && trendValue !== undefined && (
          <Group gap="xs">
            <TrendIcon size={16} color={`var(--mantine-color-${trendColor}-6)`} />
            <Text size="xs" fw={600} c={`${trendColor}.6`}>
              {trendValue > 0 ? '+' : ''}
              {trendValue.toFixed(1)}%
            </Text>
          </Group>
        )}
      </Group>
    </Card>
  );
}

export default MetricCard;
