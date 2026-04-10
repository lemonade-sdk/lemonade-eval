/**
 * Dashboard Page - Main Overview
 * Displays key metrics, recent runs, and performance trends
 */

import {
  Group,
  Grid,
  Text,
  SimpleGrid,
  Card,
  Badge,
  Button,
  Box,
  Title,
  Skeleton,
} from '@mantine/core';
import {
  IconCpu,
  IconListDetails,
  IconTrendingUp,
  IconClock,
  IconCheck,
  IconX,
} from '@tabler/icons-react';
import { Link } from 'react-router-dom';
import { useRuns, useRunStats } from '@/hooks/useRuns';
import { useModels } from '@/hooks/useModels';
import { LoadingSpinner, ErrorDisplay, StatusBadge, DataTable, MetricCard } from '@/components/common';
import { LineChart } from '@/components/charts';
import { formatDuration, formatDate } from '@/utils';
import { ColumnDef } from '@tanstack/react-table';
import type { Run } from '@/types';

export default function DashboardPage() {
  const { data: statsData, isLoading: statsLoading, error: statsError } = useRunStats();
  const { data: recentRunsData, isLoading: recentRunsLoading } = useRuns({ page: 1, per_page: 10 });
  const { data: modelsData, isLoading: modelsLoading } = useModels({ page: 1, per_page: 10 });

  const stats = statsData?.data;
  const recentRuns = recentRunsData?.data || [];
  const totalModels = modelsData?.meta?.total || 0;

  // Define columns for recent runs table
  const columns: ColumnDef<Run>[] = [
    {
      accessorKey: 'build_name',
      header: 'Run',
      cell: ({ row }) => {
        const run = row.original;
        return (
          <Link to={`/runs/${run.id}`} style={{ textDecoration: 'none' }}>
            <Text fw={500} size="sm">
              {run.build_name}
            </Text>
          </Link>
        );
      },
    },
    {
      accessorKey: 'model_id',
      header: 'Model',
      cell: ({ row }) => (
        <Text size="sm" c="dimmed">
          {row.original.model_id.slice(0, 8)}...
        </Text>
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
      header: 'Created',
      cell: ({ row }) => (
        <Text size="sm" c="dimmed">
          {formatDate(row.original.created_at)}
        </Text>
      ),
    },
  ];

  if (statsLoading || modelsLoading) {
    return <LoadingSpinner message="Loading dashboard..." />;
  }

  if (statsError) {
    return <ErrorDisplay message={statsError.message} fullScreen />;
  }

  return (
    <Box>
      <Group justify="space-between" mb="xl">
        <Title order={2}>Dashboard</Title>
        <Group>
          <Button component={Link} to="/runs" variant="outline">
            View All Runs
          </Button>
          <Button component={Link} to="/models">
            View Models
          </Button>
        </Group>
      </Group>

      {/* Summary Cards */}
      <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="md" mb="xl">
        <Card padding="lg" radius="md" withBorder>
          <Group justify="space-between">
            <div>
              <Text size="xs" c="dimmed" fw={500} tt="uppercase">
                Total Models
              </Text>
              <Text size="xl" fw={700} mt="xs">
                {totalModels}
              </Text>
            </div>
            <IconCpu size={32} color="var(--mantine-color-blue-6)" />
          </Group>
        </Card>

        <Card padding="lg" radius="md" withBorder>
          <Group justify="space-between">
            <div>
              <Text size="xs" c="dimmed" fw={500} tt="uppercase">
                Total Runs
              </Text>
              <Text size="xl" fw={700} mt="xs">
                {stats?.total_runs || 0}
              </Text>
            </div>
            <IconListDetails size={32} color="var(--mantine-color-blue-6)" />
          </Group>
        </Card>

        <Card padding="lg" radius="md" withBorder>
          <Group justify="space-between">
            <div>
              <Text size="xs" c="dimmed" fw={500} tt="uppercase">
                Completed
              </Text>
              <Text size="xl" fw={700} mt="xs" c="green.6">
                {stats?.by_status?.completed || 0}
              </Text>
            </div>
            <IconCheck size={32} color="var(--mantine-color-green-6)" />
          </Group>
        </Card>

        <Card padding="lg" radius="md" withBorder>
          <Group justify="space-between">
            <div>
              <Text size="xs" c="dimmed" fw={500} tt="uppercase">
                Failed
              </Text>
              <Text size="xl" fw={700} mt="xs" c="red.6">
                {stats?.by_status?.failed || 0}
              </Text>
            </div>
            <IconX size={32} color="var(--mantine-color-red-6)" />
          </Group>
        </Card>
      </SimpleGrid>

      {/* Status Breakdown */}
      <Grid mb="xl">
        <Grid.Col span={{ base: 12, lg: 8 }}>
          <Card padding="lg" radius="md" withBorder>
            <Title order={4} mb="md">
              Recent Runs
            </Title>
            {recentRunsLoading ? (
              <Skeleton height={200} />
            ) : recentRuns.length === 0 ? (
              <Text c="dimmed" ta="center" py="xl">
                No runs yet. Start by importing evaluation results or creating a new run.
              </Text>
            ) : (
              <DataTable
                data={recentRuns}
                columns={columns}
                enablePagination={false}
                onRowClick={(row) => (window.location.href = `/runs/${row.id}`)}
              />
            )}
          </Card>
        </Grid.Col>

        <Grid.Col span={{ base: 12, lg: 4 }}>
          <Card padding="lg" radius="md" withBorder>
            <Title order={4} mb="md">
              Run Status
            </Title>
            <Group gap="xs" wrap="wrap">
              <Badge
                variant="light"
                color="blue"
                size="lg"
                leftSection={<IconClock size={14} />}
              >
                {stats?.by_status?.running || 0} Running
              </Badge>
              <Badge
                variant="light"
                color="gray"
                size="lg"
                leftSection={<IconClock size={14} />}
              >
                {stats?.by_status?.pending || 0} Pending
              </Badge>
              <Badge
                variant="light"
                color="green"
                size="lg"
                leftSection={<IconCheck size={14} />}
              >
                {stats?.by_status?.completed || 0} Completed
              </Badge>
              <Badge
                variant="light"
                color="red"
                size="lg"
                leftSection={<IconX size={14} />}
              >
                {stats?.by_status?.failed || 0} Failed
              </Badge>
            </Group>
          </Card>
        </Grid.Col>
      </Grid>
    </Box>
  );
}
