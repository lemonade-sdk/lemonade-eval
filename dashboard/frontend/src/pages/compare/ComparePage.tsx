/**
 * Compare Page - Side-by-Side Model/Run Comparison
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
  MultiSelect,
  Divider,
  Table,
  Skeleton,
  SimpleGrid,
  SegmentedControl,
} from '@mantine/core';
import { useState, useMemo } from 'react';
import { useQueries } from '@tanstack/react-query';
import { useRuns } from '@/hooks/useRuns';
import { useCompareMetrics } from '@/hooks/useMetrics';
import { useModels } from '@/hooks/useModels';
import { LoadingSpinner, ErrorDisplay, StatusBadge, MetricCard } from '@/components/common';
import { BarChart, RadarChart } from '@/components/charts';
import { formatDateTime } from '@/utils';
import type { Model } from '@/types';
import { metricsApi } from '@/api/metrics';
import type { Run, Metric } from '@/types';

export default function ComparePage() {
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);
  const [comparisonMode, setComparisonMode] = useState<'runs' | 'models'>('runs');

  const { data: runsData, isLoading: runsLoading } = useRuns({ page: 1, per_page: 100 });
  const { data: modelsData } = useModels({ page: 1, per_page: 100 });

  const runs = runsData?.data || [];
  const models = modelsData?.data || [];

  // Create lookup maps
  const runMap = new Map(runs.map((r) => [r.id, r]));
  const modelMap = new Map(models.map((m) => [m.id, m]));

  // Get selected runs
  const selectedRuns = selectedRunIds.map((id) => runMap.get(id)).filter(Boolean) as Run[];

  // Compare metrics for selected runs — disabled in models mode
  const { data: comparisonData, isLoading: comparisonLoading } = useCompareMetrics(
    comparisonMode === 'runs' ? selectedRunIds : [],
    ['performance', 'accuracy']
  );

  // useQueries for model aggregates — only fires in models mode
  const modelAggregateQueries = useQueries({
    queries: selectedModelIds.map((modelId) => ({
      queryKey: ['metrics', 'aggregate', { model_id: modelId, category: 'performance' }],
      queryFn: () => metricsApi.getAggregateMetrics({ model_id: modelId, category: 'performance' }),
      enabled: selectedModelIds.length > 0 && comparisonMode === 'models',
    })),
  });

  const modelAggregatesLoading = modelAggregateQueries.some((q) => q.isLoading);

  const modelAggregates = modelAggregateQueries
    .map((q, i) => ({
      model: models.find((m) => m.id === selectedModelIds[i]),
      data: q.data?.data || [],
      isLoading: q.isLoading,
      isError: q.isError,
    }))
    .filter((item): item is typeof item & { model: Model } => item.model !== undefined);

  // Prepare data for radar chart
  const radarData = useMemo(() => {
    if (!comparisonData?.data) return [];

    const metrics = comparisonData.data.metrics || {};
    const allMetricNames = new Set<string>();

    Object.values(metrics).forEach((runMetrics: Metric[]) => {
      runMetrics.forEach((m: Metric) => allMetricNames.add(m.name));
    });

    return Array.from(allMetricNames).map((subject) => {
      const dataPoint: Record<string, string | number> = { subject };
      selectedRuns.forEach((run, index) => {
        const runMetrics = metrics[run.id] || [];
        const metric = runMetrics.find((m: Metric) => m.name === subject);
        dataPoint[`run_${index}`] = metric?.value_numeric ?? 0;
      });
      return dataPoint;
    });
  }, [comparisonData, selectedRuns]);

  const runOptions = runs.map((run) => ({
    value: run.id,
    label: `${run.build_name} (${run.status})`,
  }));

  return (
    <Box>
      <Group justify="space-between" mb="xl">
        <Title order={2}>{comparisonMode === 'runs' ? 'Compare Runs' : 'Compare Models'}</Title>
      </Group>

      {/* Selection Card */}
      <Card padding="lg" radius="md" withBorder mb="md">
        <Text fw={600} mb="sm">
          Select {comparisonMode === 'runs' ? 'Runs' : 'Models'} to Compare
        </Text>

        <SegmentedControl
          value={comparisonMode}
          onChange={(v) => setComparisonMode(v as 'runs' | 'models')}
          data={[
            { value: 'runs', label: 'Compare Runs' },
            { value: 'models', label: 'Compare Models' },
          ]}
          mb="sm"
        />

        {comparisonMode === 'runs' ? (
          <MultiSelect
            data={runOptions}
            value={selectedRunIds}
            onChange={(values) => setSelectedRunIds(values)}
            placeholder="Choose runs..."
            maxDropdownHeight={300}
            clearable
            searchable
            style={{ maxWidth: 600 }}
          />
        ) : (
          <MultiSelect
            data={models.map((m) => ({ value: m.id, label: m.name }))}
            value={selectedModelIds}
            onChange={(values) => setSelectedModelIds(values)}
            placeholder="Choose models to compare..."
            maxDropdownHeight={300}
            clearable
            searchable
            style={{ maxWidth: 600 }}
          />
        )}

        <Text size="xs" c="dimmed" mt="xs">
          Select 2-5 {comparisonMode === 'runs' ? 'runs' : 'models'} to compare their metrics side by side
        </Text>
      </Card>

      {(comparisonMode === 'runs' ? selectedRunIds : selectedModelIds).length < 2 ? (
        <Card padding="xl" radius="md" withBorder ta="center">
          <Text c="dimmed">
            Select at least 2 {comparisonMode === 'runs' ? 'runs' : 'models'} to see comparison results
          </Text>
        </Card>
      ) : comparisonMode === 'runs' ? (
        <>
          {/* Selected Runs Overview */}
          <Card padding="lg" radius="md" withBorder mb="md">
            <Title order={4} mb="md">
              Selected Runs
            </Title>
            <Grid>
              {selectedRuns.map((run) => (
                <Grid.Col span={{ base: 12, sm: 6, lg: 3 }} key={run.id}>
                  <Card padding="sm" radius="md" withBorder>
                    <Group justify="space-between" mb="xs">
                      <Text fw={600} size="sm" lineClamp={1}>
                        {run.build_name}
                      </Text>
                      <StatusBadge status={run.status} size="sm" />
                    </Group>
                    <Text size="xs" c="dimmed" lineClamp={1}>
                      {modelMap.get(run.model_id)?.name || 'Unknown Model'}
                    </Text>
                    <Divider my="xs" />
                    <Text size="xs" c="dimmed">
                      {run.device || '-'} / {run.backend || '-'}
                    </Text>
                    <Text size="xs" c="dimmed">
                      {formatDateTime(run.created_at)}
                    </Text>
                  </Card>
                </Grid.Col>
              ))}
            </Grid>
          </Card>

          {/* Comparison Table */}
          <Card padding="lg" radius="md" withBorder mb="md">
            <Title order={4} mb="md">
              Metrics Comparison
            </Title>
            {comparisonLoading ? (
              <Skeleton height={200} />
            ) : comparisonData?.data ? (
              <Table striped>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Metric</Table.Th>
                    {selectedRuns.map((run) => (
                      <Table.Th key={run.id} ta="center">
                        {run.build_name.length > 20 ? `${run.build_name.slice(0, 20)}…` : run.build_name}
                      </Table.Th>
                    ))}
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {/* Performance Metrics */}
                  {['seconds_to_first_token', 'token_generation_tokens_per_second', 'prefill_tokens_per_second', 'max_memory_used_gbyte'].map((metricName) => {
                    const metricValues = selectedRuns.map((run) => {
                      const metrics = comparisonData.data?.metrics?.[run.id] || [];
                      return metrics.find((m: Metric) => m.name === metricName);
                    });

                    const values = metricValues.map((m) => m?.value_numeric);
                    const maxValue = Math.max(...values.filter((v): v is number => v !== null && v !== undefined));
                    const minValue = Math.min(...values.filter((v): v is number => v !== null && v !== undefined));

                    // For metrics where lower is better (TTFT, memory), highlight min
                    // For metrics where higher is better (TPS), highlight max
                    const lowerIsBetter = ['seconds_to_first_token', 'max_memory_used_gbyte'].includes(metricName);

                    return (
                      <Table.Tr key={metricName}>
                        <Table.Td>
                          <Text fw={500} size="sm">
                            {metricName.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                          </Text>
                        </Table.Td>
                        {metricValues.map((metric, idx) => {
                          const isBest = lowerIsBetter
                            ? metric?.value_numeric === minValue
                            : metric?.value_numeric === maxValue;

                          return (
                            <Table.Td key={idx} ta="center" bg={isBest ? 'var(--mantine-color-green-light)' : undefined}>
                              <Text size="sm" fw={isBest ? 700 : 400}>
                                {metric?.value_numeric?.toFixed(3) ?? '-'}
                                {metric?.unit && ` ${metric.unit}`}
                              </Text>
                            </Table.Td>
                          );
                        })}
                      </Table.Tr>
                    );
                  })}
                </Table.Tbody>
              </Table>
            ) : (
              <Text c="dimmed" ta="center" py="xl">
                No comparison data available
              </Text>
            )}
          </Card>

          {/* Charts */}
          <Grid>
            <Grid.Col span={{ base: 12, lg: 6 }}>
              <Card padding="lg" radius="md" withBorder>
                <Title order={4} mb="md">
                  Performance Comparison
                </Title>
                {comparisonData?.data ? (
                  <BarChart
                    data={selectedRuns.map((run, idx) => {
                      const metrics = comparisonData.data?.metrics?.[run.id] || [];
                      const tpsMetric = metrics.find((m: Metric) => m.name === 'token_generation_tokens_per_second');
                      return {
                        name: `Run ${idx + 1}`,
                        value: tpsMetric?.value_numeric || 0,
                      };
                    })}
                    dataKey="value"
                    title="Tokens per Second"
                    unit="tokens/s"
                    height={250}
                    highlightBest
                  />
                ) : (
                  <Skeleton height={250} />
                )}
              </Card>
            </Grid.Col>
            <Grid.Col span={{ base: 12, lg: 6 }}>
              <Card padding="lg" radius="md" withBorder>
                <Title order={4} mb="md">
                  Multi-Metric Comparison
                </Title>
                {radarData.length > 0 ? (
                  <RadarChart
                    data={radarData}
                    dataKeys={selectedRuns.map((_, idx) => ({
                      key: `run_${idx}`,
                      name: `Run ${idx + 1}`,
                      color: idx === 0 ? 'var(--mantine-color-blue-6)' : 'var(--mantine-color-violet-6)',
                    }))}
                    height={300}
                  />
                ) : (
                  <Skeleton height={300} />
                )}
              </Card>
            </Grid.Col>
          </Grid>
        </>
      ) : (
        /* Models comparison */
        <Card padding="lg" radius="md" withBorder>
          <Title order={4} mb="md">Model Performance Comparison</Title>
          {modelAggregatesLoading ? (
            <Skeleton height={300} />
          ) : (
            <>
              <Table striped>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Metric</Table.Th>
                    {modelAggregates.map(({ model }) => (
                      <Table.Th key={model.id} ta="center">{model.name}</Table.Th>
                    ))}
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {['token_generation_tokens_per_second', 'seconds_to_first_token', 'prefill_tokens_per_second', 'max_memory_used_gbyte'].map((metricName) => {
                    const rowValues = modelAggregates.map(({ data }) => {
                      const entry = (data as any[]).find((d: any) => d.name === metricName);
                      return entry?.mean ?? null;
                    });
                    const numericValues = rowValues.filter((v): v is number => v !== null);
                    const lowerIsBetter = ['seconds_to_first_token', 'max_memory_used_gbyte'].includes(metricName);
                    const best = lowerIsBetter ? Math.min(...numericValues) : Math.max(...numericValues);
                    return (
                      <Table.Tr key={metricName}>
                        <Table.Td><Text fw={500} size="sm">{metricName.replace(/_/g, ' ')}</Text></Table.Td>
                        {rowValues.map((val, idx) => (
                          <Table.Td key={idx} ta="center" bg={val === best ? 'var(--mantine-color-green-light)' : undefined}>
                            <Text size="sm" fw={val === best ? 700 : 400}>{val !== null ? val.toFixed(3) : '-'}</Text>
                          </Table.Td>
                        ))}
                      </Table.Tr>
                    );
                  })}
                </Table.Tbody>
              </Table>
              <BarChart
                data={modelAggregates.map(({ model, data }) => {
                  const tps = (data as any[]).find((d: any) => d.name === 'token_generation_tokens_per_second');
                  return { name: model.name, value: tps?.mean ?? 0 };
                })}
                dataKey="value"
                title="Average Tokens per Second"
                unit="tok/s"
                height={250}
                highlightBest
              />
            </>
          )}
        </Card>
      )}
    </Box>
  );
}
