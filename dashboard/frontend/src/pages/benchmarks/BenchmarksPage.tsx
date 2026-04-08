/**
 * Benchmarks Page - Display Benchmark Results with Charts
 */

import {
  Group,
  Text,
  Title,
  Card,
  Grid,
  Badge,
  Box,
  Table,
  Skeleton,
  SimpleGrid,
  Paper,
  ThemeIcon,
  Divider,
  Tabs,
} from '@mantine/core';
import { IconTrophy, IconGauge, IconClock, IconDatabase } from '@tabler/icons-react';
import { useBenchmarkResults } from '@/hooks/useRuns';
import { LoadingSpinner, ErrorDisplay } from '@/components/common';
import { BarChart } from '@/components/charts';
import type { Run, Metric } from '@/types';

// Color scheme for models
const MODEL_COLORS: Record<string, string> = {
  'Qwen3.5-2B-GGUF': 'var(--mantine-color-blue-6)',
  'Qwen3.5-4B-GGUF': 'var(--mantine-color-violet-6)',
};

export default function BenchmarksPage() {
  const { data: benchmarkData, isLoading, error } = useBenchmarkResults();

  // Process benchmark data
  const processedData = benchmarkData?.data || [];

  // Find best TPS
  const tpsData = processedData.map((item: any) => {
    const tpsMetric = item.metrics?.find((m: Metric) =>
      m.name === 'token_generation_tokens_per_second'
    );
    const ttftMetric = item.metrics?.find((m: Metric) =>
      m.name === 'seconds_to_first_token'
    );
    const stdDevMetric = item.metrics?.find((m: Metric) =>
      m.name === 'std_dev_tokens_per_second'
    );

    return {
      runId: item.run.id,
      modelId: item.run.model_id,
      modelName: item.run.build_name.split('-')[0] || 'Unknown',
      tps: tpsMetric?.value_numeric || 0,
      ttft: ttftMetric?.value_numeric || 0,
      stdDev: stdDevMetric?.value_numeric || 0,
      backend: item.run.backend,
      device: item.run.device,
    };
  });

  // Sort by TPS (descending)
  const sortedByTps = [...tpsData].sort((a, b) => b.tps - a.tps);
  const winner = sortedByTps[0];

  // Group by model for comparison
  const modelComparison: Record<string, any[]> = {};
  tpsData.forEach((item) => {
    const modelName = item.modelId;
    if (!modelComparison[modelName]) {
      modelComparison[modelName] = [];
    }
    modelComparison[modelName].push(item);
  });

  // Calculate averages per model
  const modelAverages = Object.entries(modelComparison).map(([modelName, runs]) => {
    const avgTps = runs.reduce((sum, r) => sum + r.tps, 0) / runs.length;
    const avgTtft = runs.reduce((sum, r) => sum + r.ttft, 0) / runs.length;
    return {
      modelName,
      avgTps,
      avgTtft,
      runCount: runs.length,
    };
  });

  return (
    <Box>
      {/* Header */}
      <Group justify="space-between" mb="xl">
        <Group>
          <ThemeIcon size="xl" variant="gradient" gradient={{ from: 'blue', to: 'cyan' }}>
            <IconGauge size={24} />
          </ThemeIcon>
          <div>
            <Title order={2}>Benchmark Results</Title>
            <Text size="sm" c="dimmed">
              Performance comparison of evaluated models
            </Text>
          </div>
        </Group>
        <Badge size="lg" variant="light" color="blue">
          {processedData.length} Runs
        </Badge>
      </Group>

      {isLoading ? (
        <LoadingSpinner />
      ) : error ? (
        <ErrorDisplay error={error} message="Failed to load benchmark results" />
      ) : processedData.length === 0 ? (
        <Card padding="xl" radius="md" withBorder ta="center">
          <Text c="dimmed">No benchmark results available</Text>
        </Card>
      ) : (
        <>
          {/* Winner Highlight */}
          {winner && (
            <Card padding="lg" radius="md" withBorder mb="md" bg="var(--mantine-color-green-light)">
              <Group justify="space-between">
                <Group>
                  <ThemeIcon size="lg" color="green" variant="filled">
                    <IconTrophy size={20} />
                  </ThemeIcon>
                  <div>
                    <Text fw={700} size="lg">
                      Performance Leader: {winner.modelId}
                    </Text>
                    <Text size="sm" c="dimmed">
                      {winner.tps.toFixed(2)} tokens/second
                    </Text>
                  </div>
                </Group>
                <Badge size="xl" color="green" variant="filled">
                  {winner.tps.toFixed(2)} tok/s
                </Badge>
              </Group>
            </Card>
          )}

          {/* Summary Stats */}
          <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} mb="md">
            <Card padding="md" radius="md" withBorder>
              <Group justify="space-between">
                <div>
                  <Text size="xs" c="dimmed" tt="uppercase" fw={700}>
                    Total Runs
                  </Text>
                  <Text size="xl" fw={700}>
                    {processedData.length}
                  </Text>
                </div>
                <ThemeIcon size="lg" variant="light">
                  <IconDatabase size={20} />
                </ThemeIcon>
              </Group>
            </Card>

            <Card padding="md" radius="md" withBorder>
              <Group justify="space-between">
                <div>
                  <Text size="xs" c="dimmed" tt="uppercase" fw={700}>
                    Best TPS
                  </Text>
                  <Text size="xl" fw={700} c="green">
                    {winner?.tps.toFixed(2)}
                  </Text>
                </div>
                <ThemeIcon size="lg" variant="light" color="green">
                  <IconGauge size={20} />
                </ThemeIcon>
              </Group>
            </Card>

            <Card padding="md" radius="md" withBorder>
              <Group justify="space-between">
                <div>
                  <Text size="xs" c="dimmed" tt="uppercase" fw={700}>
                    Best TTFT
                  </Text>
                  <Text size="xl" fw={700} c="blue">
                    {sortedByTps.reduce((min, r) => r.ttft < min ? r.ttft : min, sortedByTps[0]?.ttft || 999).toFixed(3)}s
                  </Text>
                </div>
                <ThemeIcon size="lg" variant="light" color="blue">
                  <IconClock size={20} />
                </ThemeIcon>
              </Group>
            </Card>

            <Card padding="md" radius="md" withBorder>
              <Group justify="space-between">
                <div>
                  <Text size="xs" c="dimmed" tt="uppercase" fw={700}>
                    Models Compared
                  </Text>
                  <Text size="xl" fw={700}>
                    {Object.keys(modelComparison).length}
                  </Text>
                </div>
                <ThemeIcon size="lg" variant="light">
                  <IconTrophy size={20} />
                </ThemeIcon>
              </Group>
            </Card>
          </SimpleGrid>

          {/* TPS Comparison Chart */}
          <Card padding="lg" radius="md" withBorder mb="md">
            <Title order={4} mb="md">
              Token Generation Speed Comparison
            </Title>
            <BarChart
              data={sortedByTps.map((item) => ({
                name: item.modelId.replace('-GGUF', '').split('-').pop() || item.modelId,
                value: item.tps,
                color: MODEL_COLORS[item.modelId] || 'var(--mantine-color-gray-6)',
              }))}
              dataKey="value"
              title="Tokens per Second (higher is better)"
              unit="tok/s"
              height={300}
              highlightBest
            />
          </Card>

          {/* Detailed Comparison Table */}
          <Card padding="lg" radius="md" withBorder mb="md">
            <Title order={4} mb="md">
              Detailed Results
            </Title>
            <Table striped highlightOnHover>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Model</Table.Th>
                  <Table.Th ta="right">TPS (tok/s)</Table.Th>
                  <Table.Th ta="right">TTFT (s)</Table.Th>
                  <Table.Th ta="right">Std Dev</Table.Th>
                  <Table.Th>Backend</Table.Th>
                  <Table.Th>Device</Table.Th>
                  <Table.Th>Run ID</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {sortedByTps.map((item, index) => (
                  <Table.Tr
                    key={item.runId}
                    bg={index === 0 ? 'var(--mantine-color-green-light)' : undefined}
                  >
                    <Table.Td>
                      <Text fw={index === 0 ? 700 : 400}>
                        {item.modelId}
                      </Text>
                    </Table.Td>
                    <Table.Td ta="right">
                      <Text fw={index === 0 ? 700 : 400}>
                        {item.tps.toFixed(2)}
                      </Text>
                    </Table.Td>
                    <Table.Td ta="right">
                      {item.ttft.toFixed(3)}
                    </Table.Td>
                    <Table.Td ta="right">
                      {item.stdDev.toFixed(2)}
                    </Table.Td>
                    <Table.Td>
                      <Badge size="sm" variant="light">
                        {item.backend || '-'}
                      </Badge>
                    </Table.Td>
                    <Table.Td>
                      <Badge size="sm" variant="light">
                        {item.device || '-'}
                      </Badge>
                    </Table.Td>
                    <Table.Td>
                      <Text size="xs" c="dimmed" style={{ maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {item.runId.slice(0, 8)}...
                      </Text>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Card>

          {/* Model Averages Comparison */}
          <Card padding="lg" radius="md" withBorder>
            <Title order={4} mb="md">
              Model Averages
            </Title>
            <SimpleGrid cols={{ base: 1, sm: 2 }}>
              {modelAverages.map((model) => (
                <Card key={model.modelName} padding="md" radius="md" withBorder>
                  <Group justify="space-between" mb="sm">
                    <Text fw={600}>{model.modelName}</Text>
                    <Badge variant="light">{model.runCount} runs</Badge>
                  </Group>
                  <Divider mb="sm" />
                  <Group justify="space-between">
                    <div>
                      <Text size="xs" c="dimmed">Avg TPS</Text>
                      <Text size="lg" fw={700} c="green">
                        {model.avgTps.toFixed(2)} tok/s
                      </Text>
                    </div>
                    <div>
                      <Text size="xs" c="dimmed">Avg TTFT</Text>
                      <Text size="lg" fw={700} c="blue">
                        {model.avgTtft.toFixed(3)}s
                      </Text>
                    </div>
                  </Group>
                </Card>
              ))}
            </SimpleGrid>
          </Card>
        </>
      )}
    </Box>
  );
}
