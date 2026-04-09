/**
 * Accuracy Page
 *
 * Displays MMLU subject-level accuracy, HumanEval run history,
 * and perplexity trend data, filterable by model.
 */

import { useState, useMemo, useEffect } from 'react';
import { Box, Title, Group, Tabs, Select, Text, Card, ScrollArea, Skeleton } from '@mantine/core';
import { IconTargetArrow } from '@tabler/icons-react';
import { useRuns } from '@/hooks/useRuns';
import { useModels } from '@/hooks/useModels';
import { useMetrics, useMetricTrends } from '@/hooks/useMetrics';
import { LoadingSpinner, MetricCard, DataTable } from '@/components/common';
import { BarChart, LineChart } from '@/components/charts';
import { ColumnDef } from '@tanstack/react-table';
import type { Run, Metric } from '@/types';

export default function AccuracyPage() {
  const [activeTab, setActiveTab] = useState<string>('mmlu');
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [mmluRunId, setMmluRunId] = useState<string | null>(null);

  const { data: modelsData } = useModels({ page: 1, per_page: 100 });
  const models = modelsData?.data || [];

  // Runs for each accuracy type
  const { data: mmluRunsData, isLoading: mmluLoading } = useRuns({ run_type: 'accuracy-mmlu', model_id: selectedModelId ?? undefined, page: 1, per_page: 50 });
  const { data: humanEvalRunsData, isLoading: heLoading } = useRuns({ run_type: 'accuracy-humaneval', model_id: selectedModelId ?? undefined, page: 1, per_page: 50 });

  const mmluRuns = mmluRunsData?.data || [];
  const humanEvalRuns = humanEvalRunsData?.data || [];

  // Reset run selection when model changes (W8-3)
  useEffect(() => {
    setMmluRunId(null);
  }, [selectedModelId]);

  // Auto-select latest MMLU run when data loads
  const latestMmluRunId = mmluRunId ?? mmluRuns[0]?.id ?? null;

  // Fetch metrics for selected MMLU run
  const { metrics: mmluMetrics, isLoading: mmluMetricsLoading } = useMetrics(
    latestMmluRunId ? { run_id: latestMmluRunId, category: 'accuracy' } : undefined
  );

  // Filter and sort MMLU subject metrics
  const mmluSubjectMetrics = useMemo(() => {
    return mmluMetrics
      .filter((m) => m.name.startsWith('mmlu_') && m.name.endsWith('_accuracy') && m.name !== 'average_mmlu_accuracy')
      .map((m) => ({
        name: m.name.replace(/^mmlu_/, '').replace(/_accuracy$/, '').replace(/_/g, ' '),
        value: m.value_numeric ?? 0,
      }))
      .sort((a, b) => b.value - a.value);
  }, [mmluMetrics]);

  const averageMmlu = mmluMetrics.find((m) => m.name === 'average_mmlu_accuracy')?.value_numeric;

  // Perplexity trend — only fetches when tab is active and a model is selected (W8-1)
  const { trends: perplexityTrends, isLoading: perplexityLoading } = useMetricTrends(
    selectedModelId ?? '',
    'perplexity',
    100,
    activeTab === 'perplexity'
  );

  const modelOptions = models.map((m) => ({ value: m.id, label: m.name }));
  const mmluRunOptions = mmluRuns.map((r) => ({ value: r.id, label: r.build_name }));

  // HumanEval columns
  const heColumns: ColumnDef<Run>[] = [
    { accessorKey: 'build_name', header: 'Run' },
    { accessorKey: 'status', header: 'Status', cell: ({ row }) => <Text size="sm">{row.original.status}</Text> },
    { accessorKey: 'created_at', header: 'Date', cell: ({ row }) => <Text size="sm" c="dimmed">{new Date(row.original.created_at).toLocaleDateString()}</Text> },
  ];

  return (
    <Box>
      <Group justify="space-between" mb="xl">
        <Group>
          <IconTargetArrow size={28} color="var(--mantine-color-blue-6)" />
          <Title order={2}>Accuracy</Title>
        </Group>
        <Select
          placeholder="All models"
          data={modelOptions}
          value={selectedModelId}
          onChange={setSelectedModelId}
          clearable
          searchable
          w={250}
        />
      </Group>

      <Tabs value={activeTab} onChange={(v) => v && setActiveTab(v)}>
        <Tabs.List mb="md">
          <Tabs.Tab value="mmlu">MMLU</Tabs.Tab>
          <Tabs.Tab value="humaneval">HumanEval</Tabs.Tab>
          <Tabs.Tab value="perplexity">Perplexity</Tabs.Tab>
        </Tabs.List>

        {/* MMLU Tab */}
        <Tabs.Panel value="mmlu">
          {mmluLoading ? <LoadingSpinner /> : mmluRuns.length === 0 ? (
            <Card padding="xl" withBorder ta="center"><Text c="dimmed">No MMLU runs found.</Text></Card>
          ) : (
            <>
              <Group mb="md" justify="space-between">
                <Select
                  label="Run"
                  data={mmluRunOptions}
                  value={latestMmluRunId}
                  onChange={(v) => setMmluRunId(v)}
                  w={350}
                />
                {averageMmlu !== undefined && (
                  <MetricCard label="Average MMLU Accuracy" value={averageMmlu.toFixed(1)} unit="%" />
                )}
              </Group>
              {mmluMetricsLoading ? <Skeleton height={700} /> : mmluSubjectMetrics.length === 0 ? (
                <Text c="dimmed" ta="center" py="xl">No subject-level data available.</Text>
              ) : (
                <Card withBorder padding="md">
                  <Text size="sm" c="dimmed" mb="sm">{mmluSubjectMetrics.length} subjects — sorted by accuracy (highest first)</Text>
                  <ScrollArea h={700}>
                    <BarChart
                      data={mmluSubjectMetrics}
                      dataKey="value"
                      nameKey="name"
                      layout="vertical"
                      height={Math.max(700, mmluSubjectMetrics.length * 28)}
                      unit="%"
                      showLegend={false}
                      highlightBest={false}
                    />
                  </ScrollArea>
                </Card>
              )}
            </>
          )}
        </Tabs.Panel>

        {/* HumanEval Tab */}
        <Tabs.Panel value="humaneval">
          {heLoading ? <LoadingSpinner /> : humanEvalRuns.length === 0 ? (
            <Card padding="xl" withBorder ta="center"><Text c="dimmed">No HumanEval runs found.</Text></Card>
          ) : (
            <Card withBorder padding="lg">
              <Title order={4} mb="md">HumanEval Runs</Title>
              <DataTable data={humanEvalRuns} columns={heColumns} enablePagination={false} />
            </Card>
          )}
        </Tabs.Panel>

        {/* Perplexity Tab */}
        <Tabs.Panel value="perplexity">
          {perplexityLoading ? <LoadingSpinner /> : perplexityTrends.length === 0 ? (
            <Card padding="xl" withBorder ta="center"><Text c="dimmed">No perplexity data found.</Text></Card>
          ) : (
            <Card withBorder padding="lg">
              <Title order={4} mb="md">Perplexity Trend</Title>
              <LineChart
                data={perplexityTrends.map((t) => ({ name: t.run_name, value: t.value }))}
                dataKey="value"
                nameKey="name"
                height={300}
              />
            </Card>
          )}
        </Tabs.Panel>
      </Tabs>
    </Box>
  );
}
