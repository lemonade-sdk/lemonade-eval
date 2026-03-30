/**
 * Run Detail Page - View Run Details and Metrics
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
  Tabs,
  Code,
  ScrollArea,
} from '@mantine/core';
import { useParams, Link } from 'react-router-dom';
import { IconArrowLeft, IconCode, IconTable } from '@tabler/icons-react';
import { useRun, useRunMetrics } from '@/hooks/useRuns';
import { useModel } from '@/hooks/useModels';
import { useWebSocket } from '@/hooks/useWebSocket';
import { LoadingSpinner, ErrorDisplay, StatusBadge, MetricCard, DataTable } from '@/components/common';
import { BarChart, LineChart } from '@/components/charts';
import { formatDateTime, formatDuration } from '@/utils';
import { ColumnDef } from '@tanstack/react-table';
import type { Metric } from '@/types';

export default function RunDetailPage() {
  const { id } = useParams<{ id: string }>();

  if (!id) {
    return <ErrorDisplay message="Run ID is required" fullScreen />;
  }

  const { data: runData, isLoading: runLoading, error: runError } = useRun(id, true);
  const { data: metricsData, isLoading: metricsLoading } = useRunMetrics(id);
  const { isConnected } = useWebSocket(id);
  const { data: modelData } = useModel(runData?.data?.model_id || '', !!runData?.data?.model_id);

  const run = runData?.data;
  const metrics = metricsData?.data || [];
  const model = modelData?.data;

  // Group metrics by category
  const performanceMetrics = metrics.filter((m) => m.category === 'performance');
  const accuracyMetrics = metrics.filter((m) => m.category === 'accuracy');

  // Prepare chart data for iteration values
  const iterationData = performanceMetrics
    .filter((m) => m.iteration_values && m.iteration_values.length > 0)
    .map((m, i) => {
      const maxLength = Math.max(...performanceMetrics.filter(p => p.iteration_values).map(p => p.iteration_values?.length || 0));
      return Array.from({ length: maxLength }, (_, idx) => ({
        iteration: idx + 1,
        [m.name]: m.iteration_values?.[idx] || null,
      }));
    })
    .flat();

  const columns: ColumnDef<Metric>[] = [
    {
      accessorKey: 'name',
      header: 'Metric',
      cell: ({ row }) => (
        <Text fw={500} size="sm">
          {row.original.display_name || row.original.name}
        </Text>
      ),
    },
    {
      accessorKey: 'category',
      header: 'Category',
      cell: ({ row }) => (
        <Badge variant="light" color={row.original.category === 'performance' ? 'blue' : 'green'} size="sm">
          {row.original.category}
        </Badge>
      ),
    },
    {
      accessorKey: 'value_numeric',
      header: 'Value',
      cell: ({ row }) => (
        <Text fw={600} size="sm">
          {row.original.value_numeric?.toLocaleString() ?? row.original.value_text ?? '-'}
          {row.original.unit && ` ${row.original.unit}`}
        </Text>
      ),
    },
    {
      accessorKey: 'mean_value',
      header: 'Mean',
      cell: ({ row }) => (
        <Text size="sm" c="dimmed">
          {row.original.mean_value?.toFixed(3) ?? '-'}
        </Text>
      ),
    },
    {
      accessorKey: 'std_dev',
      header: 'Std Dev',
      cell: ({ row }) => (
        <Text size="sm" c="dimmed">
          {row.original.std_dev?.toFixed(4) ?? '-'}
        </Text>
      ),
    },
  ];

  if (runLoading) {
    return <LoadingSpinner message="Loading run details..." />;
  }

  if (runError || !run) {
    return <ErrorDisplay message={runError?.message || 'Run not found'} fullScreen />;
  }

  return (
    <Box>
      {/* Header */}
      <Group justify="space-between" mb="xl">
        <Group>
          <Button
            component={Link}
            to="/runs"
            variant="outline"
            leftSection={<IconArrowLeft size={16} />}
          >
            Back to Runs
          </Button>
          <Group gap="xs">
            {isConnected && (
              <Badge size="sm" color="green" variant="dot">
                Live
              </Badge>
            )}
          </Group>
        </Group>
        <StatusBadge status={run.status} size="lg" />
      </Group>

      {/* Run Info */}
      <Card padding="lg" radius="md" withBorder mb="md">
        <Group justify="space-between" wrap="wrap" mb="md">
          <div>
            <Title order={2}>{run.build_name}</Title>
            <Text c="dimmed" mt="xs">
              ID: {run.id}
            </Text>
          </div>
          <Group>
            <Badge variant="light" size="lg">
              {run.run_type}
            </Badge>
            {model && (
              <Badge component={Link} to={`/models/${model.id}`} variant="light" color="blue" size="lg">
                {model.name}
              </Badge>
            )}
          </Group>
        </Group>

        <Divider my="md" />

        <Grid>
          <Grid.Col span={{ base: 6, sm: 3 }}>
            <Text size="xs" c="dimmed" fw={500} tt="uppercase">
              Status
            </Text>
            <Box mt="xs">
              <StatusBadge status={run.status} />
            </Box>
          </Grid.Col>
          <Grid.Col span={{ base: 6, sm: 3 }}>
            <Text size="xs" c="dimmed" fw={500} tt="uppercase">
              Device
            </Text>
            <Text size="lg" fw={600} mt="xs">
              {run.device || '-'}
            </Text>
          </Grid.Col>
          <Grid.Col span={{ base: 6, sm: 3 }}>
            <Text size="xs" c="dimmed" fw={500} tt="uppercase">
              Backend
            </Text>
            <Text size="lg" fw={600} mt="xs">
              {run.backend || '-'}
            </Text>
          </Grid.Col>
          <Grid.Col span={{ base: 6, sm: 3 }}>
            <Text size="xs" c="dimmed" fw={500} tt="uppercase">
              Dtype
            </Text>
            <Text size="lg" fw={600} mt="xs">
              {run.dtype || '-'}
            </Text>
          </Grid.Col>
        </Grid>

        <Grid mt="md">
          <Grid.Col span={{ base: 6, sm: 3 }}>
            <Text size="xs" c="dimmed" fw={500} tt="uppercase">
              Started
            </Text>
            <Text size="sm" mt="xs">
              {run.started_at ? formatDateTime(run.started_at) : '-'}
            </Text>
          </Grid.Col>
          <Grid.Col span={{ base: 6, sm: 3 }}>
            <Text size="xs" c="dimmed" fw={500} tt="uppercase">
              Completed
            </Text>
            <Text size="sm" mt="xs">
              {run.completed_at ? formatDateTime(run.completed_at) : '-'}
            </Text>
          </Grid.Col>
          <Grid.Col span={{ base: 6, sm: 3 }}>
            <Text size="xs" c="dimmed" fw={500} tt="uppercase">
              Duration
            </Text>
            <Text size="sm" mt="xs" fw={600}>
              {run.duration_seconds ? formatDuration(run.duration_seconds) : '-'}
            </Text>
          </Grid.Col>
          <Grid.Col span={{ base: 6, sm: 3 }}>
            <Text size="xs" c="dimmed" fw={500} tt="uppercase">
              Lemonade Version
            </Text>
            <Text size="sm" mt="xs">
              {run.lemonade_version || '-'}
            </Text>
          </Grid.Col>
        </Grid>

        {run.status_message && (
          <Box mt="md">
            <Text size="xs" c="dimmed" fw={500} tt="uppercase">
              Status Message
            </Text>
            <Text size="sm" mt="xs" c={run.status === 'failed' ? 'red' : undefined}>
              {run.status_message}
            </Text>
          </Box>
        )}
      </Card>

      {/* Key Metrics */}
      {performanceMetrics.length > 0 && (
        <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="md" mb="md">
          {performanceMetrics
            .filter((m) => ['seconds_to_first_token', 'token_generation_tokens_per_second', 'prefill_tokens_per_second', 'max_memory_used_gbyte'].includes(m.name))
            .map((metric) => (
              <MetricCard
                key={metric.id}
                label={metric.display_name || metric.name.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                value={metric.value_numeric?.toFixed(3) ?? '-'}
                unit={metric.unit}
                highlight={metric.name === 'token_generation_tokens_per_second'}
              />
            ))}
        </SimpleGrid>
      )}

      <Tabs defaultValue="metrics">
        <Tabs.List>
          <Tabs.Tab value="metrics" leftSection={<IconTable size={14} />}>
            Metrics Table
          </Tabs.Tab>
          <Tabs.Tab value="charts" leftSection={<IconCode size={14} />}>
            Charts
          </Tabs.Tab>
          {run.error_log && (
            <Tabs.Tab value="logs">
              Error Log
            </Tabs.Tab>
          )}
        </Tabs.List>

        <Tabs.Panel value="metrics" pt="md">
          <Grid>
            <Grid.Col span={{ base: 12, lg: 6 }}>
              <Card padding="lg" radius="md" withBorder>
                <Title order={4} mb="md">
                  Performance Metrics
                </Title>
                {performanceMetrics.length === 0 ? (
                  <Text c="dimmed" ta="center" py="xl">
                    No performance metrics available
                  </Text>
                ) : (
                  <DataTable
                    data={performanceMetrics}
                    columns={columns}
                    enablePagination={false}
                  />
                )}
              </Card>
            </Grid.Col>
            <Grid.Col span={{ base: 12, lg: 6 }}>
              <Card padding="lg" radius="md" withBorder>
                <Title order={4} mb="md">
                  Accuracy Metrics
                </Title>
                {accuracyMetrics.length === 0 ? (
                  <Text c="dimmed" ta="center" py="xl">
                    No accuracy metrics available
                  </Text>
                ) : (
                  <DataTable
                    data={accuracyMetrics}
                    columns={columns}
                    enablePagination={false}
                  />
                )}
              </Card>
            </Grid.Col>
          </Grid>
        </Tabs.Panel>

        <Tabs.Panel value="charts" pt="md">
          <Grid>
            <Grid.Col span={{ base: 12 }}>
              <Card padding="lg" radius="md" withBorder>
                <Title order={4} mb="md">
                  Performance Over Iterations
                </Title>
                {iterationData.length === 0 ? (
                  <Text c="dimmed" ta="center" py="xl">
                    No iteration data available
                  </Text>
                ) : (
                  <LineChart
                    data={iterationData}
                    dataKey="iteration"
                    multipleLines={performanceMetrics
                      .filter((m) => m.iteration_values && m.iteration_values.length > 0)
                      .map((m, i) => ({
                        key: m.name,
                        name: m.display_name || m.name,
                        color: i === 0 ? 'var(--mantine-color-blue-6)' : 'var(--mantine-color-violet-6)',
                      }))}
                    height={300}
                  />
                )}
              </Card>
            </Grid.Col>
          </Grid>
        </Tabs.Panel>

        {run.error_log && (
          <Tabs.Panel value="logs" pt="md">
            <Card padding="lg" radius="md" withBorder>
              <Title order={4} mb="md">
                Error Log
              </Title>
              <ScrollArea h={300}>
                <Code block c="red">
                  {run.error_log}
                </Code>
              </ScrollArea>
            </Card>
          </Tabs.Panel>
        )}
      </Tabs>
    </Box>
  );
}
