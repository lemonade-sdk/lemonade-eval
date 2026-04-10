/**
 * Model Detail Page
 */

import {
  Group,
  Text,
  Title,
  Card,
  Grid,
  Badge,
  Button,
  Box,
  Skeleton,
  Divider,
  SimpleGrid,
  Anchor,
  Select,
} from '@mantine/core';
import { useParams, Link } from 'react-router-dom';
import { useState } from 'react';
import { IconArrowLeft, IconExternalLink } from '@tabler/icons-react';
import { useModel, useModelRuns } from '@/hooks/useModels';
import { useRuns } from '@/hooks/useRuns';
import { useMetricTrends } from '@/hooks/useMetrics';
import { LoadingSpinner, ErrorDisplay, StatusBadge, DataTable, MetricCard } from '@/components/common';
import { BarChart, LineChart } from '@/components/charts';
import { formatDateTime, extractModelFamily, parseQuantization } from '@/utils';
import { ColumnDef } from '@tanstack/react-table';
import type { Run } from '@/types';

export default function ModelDetailPage() {
  const { id } = useParams<{ id: string }>();

  if (!id) {
    return <ErrorDisplay message="Model ID is required" fullScreen />;
  }

  const { data: modelData, isLoading: modelLoading, error } = useModel(id);
  const { data: runsData, isLoading: runsLoading } = useModelRuns(id, 10);

  const model = modelData?.data;
  const runs = runsData?.data || [];

  const [trendMetric, setTrendMetric] = useState('token_generation_tokens_per_second');
  const { trends, isLoading: trendsLoading } = useMetricTrends(id, trendMetric);

  const trendChartData = trends.map((t) => ({
    name: t.run_name,
    value: t.value,
  }));

  // Prepare data for chart - run types distribution
  const runTypeCounts = runs.reduce((acc, run) => {
    acc[run.run_type] = (acc[run.run_type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const chartData = Object.entries(runTypeCounts).map(([name, value]) => ({ name, value }));

  const columns: ColumnDef<Run>[] = [
    {
      accessorKey: 'build_name',
      header: 'Run',
      cell: ({ row }) => (
        <Link to={`/runs/${row.original.id}`} style={{ textDecoration: 'none' }}>
          <Text fw={500} size="sm">
            {row.original.build_name}
          </Text>
        </Link>
      ),
    },
    {
      accessorKey: 'run_type',
      header: 'Type',
      cell: ({ row }) => (
        <Badge variant="light" size="sm">
          {row.original.run_type}
        </Badge>
      ),
    },
    {
      accessorKey: 'status',
      header: 'Status',
      cell: ({ row }) => <StatusBadge status={row.original.status} size="sm" />,
    },
    {
      accessorKey: 'device',
      header: 'Device',
      cell: ({ row }) => (
        <Text size="sm" c="dimmed">
          {row.original.device || '-'}
        </Text>
      ),
    },
    {
      accessorKey: 'backend',
      header: 'Backend',
      cell: ({ row }) => (
        <Text size="sm" c="dimmed">
          {row.original.backend || '-'}
        </Text>
      ),
    },
    {
      accessorKey: 'created_at',
      header: 'Date',
      cell: ({ row }) => (
        <Text size="sm" c="dimmed">
          {formatDateTime(row.original.created_at)}
        </Text>
      ),
    },
  ];

  if (modelLoading) {
    return <LoadingSpinner message="Loading model..." />;
  }

  if (error || !model) {
    return <ErrorDisplay message={error?.message || 'Model not found'} fullScreen />;
  }

  return (
    <Box>
      {/* Header */}
      <Group mb="xl">
        <Button
          component={Link}
          to="/models"
          variant="outline"
          leftSection={<IconArrowLeft size={16} />}
        >
          Back to Models
        </Button>
      </Group>

      {/* Model Info */}
      <Card padding="lg" radius="md" withBorder mb="md">
        <Group justify="space-between" wrap="wrap" mb="md">
          <div>
            <Title order={2}>{model.name}</Title>
            <Text c="dimmed" mt="xs">
              {model.checkpoint}
            </Text>
          </div>
          <Group>
            <Badge variant="light" color="blue" size="lg">
              {model.model_type.toUpperCase()}
            </Badge>
            <Badge variant="light" size="lg">
              {model.family || extractModelFamily(model.checkpoint)}
            </Badge>
            {parseQuantization(model.checkpoint) && (
              <Badge variant="outline" size="lg">
                {parseQuantization(model.checkpoint)}
              </Badge>
            )}
          </Group>
        </Group>

        <Divider my="md" />

        <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="md">
          <Box>
            <Text size="xs" c="dimmed" fw={500} tt="uppercase">
              Parameters
            </Text>
            <Text size="lg" fw={600}>
              {model.parameters
                ? model.parameters >= 1e9
                  ? `${(model.parameters / 1e9).toFixed(1)}B`
                  : `${(model.parameters / 1e6).toFixed(0)}M`
                : 'Unknown'}
            </Text>
          </Box>
          <Box>
            <Text size="xs" c="dimmed" fw={500} tt="uppercase">
              Max Context
            </Text>
            <Text size="lg" fw={600}>
              {model.max_context_length ? `${model.max_context_length.toLocaleString()} tokens` : 'Unknown'}
            </Text>
          </Box>
          <Box>
            <Text size="xs" c="dimmed" fw={500} tt="uppercase">
              Architecture
            </Text>
            <Text size="lg" fw={600}>
              {model.architecture || 'Unknown'}
            </Text>
          </Box>
          <Box>
            <Text size="xs" c="dimmed" fw={500} tt="uppercase">
              License
            </Text>
            <Text size="lg" fw={600}>
              {model.license_type || 'Unknown'}
            </Text>
          </Box>
        </SimpleGrid>

        {model.hf_repo && (
          <Box mt="md">
            <Text size="xs" c="dimmed" fw={500} tt="uppercase" mb="xs">
              HuggingFace Repository
            </Text>
            <Anchor
              href={`https://huggingface.co/${model.hf_repo}`}
              target="_blank"
              leftSection={<IconExternalLink size={14} />}
            >
              {model.hf_repo}
            </Anchor>
          </Box>
        )}
      </Card>

      <Grid>
        <Grid.Col span={{ base: 12, lg: 8 }}>
          <Card padding="lg" radius="md" withBorder>
            <Title order={4} mb="md">
              Recent Runs
            </Title>
            {runsLoading ? (
              <Skeleton height={300} />
            ) : runs.length === 0 ? (
              <Text c="dimmed" ta="center" py="xl">
                No runs for this model yet.
              </Text>
            ) : (
              <DataTable
                data={runs}
                columns={columns}
                enablePagination={false}
              />
            )}
          </Card>
        </Grid.Col>

        <Grid.Col span={{ base: 12, lg: 4 }}>
          <Card padding="lg" radius="md" withBorder>
            <Title order={4} mb="md">
              Run Distribution
            </Title>
            {chartData.length === 0 ? (
              <Text c="dimmed" ta="center" py="xl">
                No data available
              </Text>
            ) : (
              <BarChart
                data={chartData}
                dataKey="value"
                height={250}
                showLegend={false}
                showGrid={false}
              />
            )}
          </Card>
        </Grid.Col>
      </Grid>

      {/* Performance Trend */}
      <Card padding="lg" radius="md" withBorder mt="md">
        <Group justify="space-between" mb="md">
          <Title order={4}>Performance Trend</Title>
          <Select
            size="sm"
            value={trendMetric}
            onChange={(v) => v && setTrendMetric(v)}
            data={[
              { value: 'token_generation_tokens_per_second', label: 'Tokens / Second' },
              { value: 'seconds_to_first_token', label: 'Time to First Token (s)' },
              { value: 'prefill_tokens_per_second', label: 'Prefill Speed (tok/s)' },
              { value: 'max_memory_used_gbyte', label: 'Peak Memory (GB)' },
            ]}
            w={220}
          />
        </Group>
        {trendsLoading ? (
          <Skeleton height={250} />
        ) : trendChartData.length === 0 ? (
          <Text c="dimmed" ta="center" py="xl">
            No trend data available for this metric.
          </Text>
        ) : (
          <LineChart
            data={trendChartData}
            dataKey="value"
            nameKey="name"
            height={250}
          />
        )}
      </Card>
    </Box>
  );
}
