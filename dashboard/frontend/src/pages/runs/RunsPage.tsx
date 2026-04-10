/**
 * Runs Page - List and Filter Evaluation Runs
 */

import {
  Group,
  Text,
  Title,
  Card,
  Grid,
  Badge,
  TextInput,
  Select,
  Box,
  Skeleton,
  MultiSelect,
  Divider,
} from '@mantine/core';
import { useState } from 'react';
import { IconSearch, IconFilter } from '@tabler/icons-react';
import { useRuns } from '@/hooks/useRuns';
import { useModels } from '@/hooks/useModels';
import { LoadingSpinner, ErrorDisplay, DataTable, StatusBadge } from '@/components/common';
import { formatDateTime, formatDuration } from '@/utils';
import { ColumnDef } from '@tanstack/react-table';
import type { Run, RunStatus, RunType } from '@/types';

export default function RunsPage() {
  const [search, setSearch] = useState('');
  const [selectedStatus, setSelectedStatus] = useState<string[]>([]);
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null);
  const [selectedBackend, setSelectedBackend] = useState<string | null>(null);

  const { data: runsData, isLoading, error } = useRuns({
    page: 1,
    per_page: 50,
    status: selectedStatus.length > 0 ? selectedStatus.join(',') : null,
    run_type: selectedType,
    device: selectedDevice,
    backend: selectedBackend,
  });

  const { data: modelsData } = useModels({ page: 1, per_page: 100 });

  const runs = runsData?.data || [];
  const models = modelsData?.data || [];

  // Create model lookup map
  const modelMap = new Map(models.map((m) => [m.id, m]));

  const columns: ColumnDef<Run>[] = [
    {
      accessorKey: 'build_name',
      header: 'Run',
      cell: ({ row }) => {
        const run = row.original;
        return (
          <div>
            <Text fw={500} size="sm" component="a" href={`/runs/${run.id}`} style={{ textDecoration: 'none', color: 'inherit' }}>
              {run.build_name}
            </Text>
            <Text size="xs" c="dimmed">
              {modelMap.get(run.model_id)?.name || run.model_id.slice(0, 8)}
            </Text>
          </div>
        );
      },
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
        <Badge variant="outline" size="sm">
          {row.original.device || '-'}
        </Badge>
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
      accessorKey: 'duration_seconds',
      header: 'Duration',
      cell: ({ row }) => (
        <Text size="sm">
          {row.original.duration_seconds ? formatDuration(row.original.duration_seconds) : '-'}
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

  if (error) {
    return <ErrorDisplay message={error.message} fullScreen />;
  }

  return (
    <Box>
      <Group justify="space-between" mb="xl">
        <Title order={2}>Evaluation Runs</Title>
      </Group>

      {/* Filters */}
      <Card padding="md" radius="md" withBorder mb="md">
        <Grid>
          <Grid.Col span={{ base: 12, md: 6, lg: 3 }}>
            <TextInput
              placeholder="Search runs..."
              leftSection={<IconSearch size={16} />}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              size="sm"
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, sm: 6, lg: 2 }}>
            <MultiSelect
              placeholder="All statuses"
              data={[
                { value: 'pending', label: 'Pending' },
                { value: 'running', label: 'Running' },
                { value: 'completed', label: 'Completed' },
                { value: 'failed', label: 'Failed' },
                { value: 'cancelled', label: 'Cancelled' },
              ]}
              value={selectedStatus}
              onChange={setSelectedStatus}
              size="sm"
              clearable
              leftSection={<IconFilter size={14} />}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, sm: 6, lg: 2 }}>
            <Select
              placeholder="All types"
              data={[
                { value: 'benchmark', label: 'Benchmark' },
                { value: 'accuracy-mmlu', label: 'MMLU' },
                { value: 'accuracy-humaneval', label: 'HumanEval' },
                { value: 'lm-eval', label: 'LM Eval' },
                { value: 'perplexity', label: 'Perplexity' },
              ]}
              value={selectedType}
              onChange={setSelectedType}
              size="sm"
              clearable
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, sm: 6, lg: 2 }}>
            <Select
              placeholder="All devices"
              data={[
                { value: 'cpu', label: 'CPU' },
                { value: 'gpu', label: 'GPU' },
                { value: 'npu', label: 'NPU' },
                { value: 'hybrid', label: 'Hybrid' },
              ]}
              value={selectedDevice}
              onChange={setSelectedDevice}
              size="sm"
              clearable
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, sm: 6, lg: 2 }}>
            <Select
              placeholder="All backends"
              data={[
                { value: 'llamacpp', label: 'LLamaCpp' },
                { value: 'ort', label: 'ORT' },
                { value: 'flm', label: 'FLM' },
              ]}
              value={selectedBackend}
              onChange={setSelectedBackend}
              size="sm"
              clearable
            />
          </Grid.Col>
        </Grid>
      </Card>

      {/* Runs Table */}
      <Card padding="lg" radius="md" withBorder>
        {isLoading ? (
          <Skeleton height={400} />
        ) : runs.length === 0 ? (
          <Text c="dimmed" ta="center" py="xl">
            No runs found. Start by importing evaluation results.
          </Text>
        ) : (
          <DataTable
            data={runs}
            columns={columns}
            onRowClick={(row) => (window.location.href = `/runs/${row.id}`)}
          />
        )}
      </Card>
    </Box>
  );
}
