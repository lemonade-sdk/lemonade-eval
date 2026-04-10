/**
 * Benchmarks Page - Display Benchmark Results with Charts
 */

import {
  Group,
  Text,
  Title,
  Card,
  Badge,
  Box,
  Table,
  SimpleGrid,
  ThemeIcon,
  Divider,
} from '@mantine/core';
import { IconTrophy, IconGauge, IconClock, IconDatabase } from '@tabler/icons-react';
import { useBenchmarkResults } from '@/hooks/useRuns';
import { LoadingSpinner, ErrorDisplay } from '@/components/common';
import { BarChart, LineChart } from '@/components/charts';
import type { Metric } from '@/types';

const PROMPT_LENGTHS = [64, 128, 256] as const;

const SWEEP_PALETTE = [
  'var(--mantine-color-blue-6)',
  'var(--mantine-color-violet-6)',
  'var(--mantine-color-teal-6)',
  'var(--mantine-color-orange-6)',
  'var(--mantine-color-pink-6)',
  'var(--mantine-color-green-6)',
  'var(--mantine-color-yellow-7)',
  'var(--mantine-color-cyan-6)',
];

function buildRunLabel(run: { build_name: string; backend?: string | null }): string {
  const base = run.build_name.length > 18 ? run.build_name.slice(0, 18) : run.build_name;
  return run.backend ? `${base} (${run.backend})` : base;
}

export default function BenchmarksPage() {
  const { results: processedData, isLoading, error } = useBenchmarkResults();

  // Find best TPS + per-prompt-length values
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
      tps: tpsMetric?.value_numeric ?? 0,
      ttft: ttftMetric?.value_numeric ?? 0,
      stdDev: stdDevMetric?.value_numeric ?? 0,
      backend: item.run.backend,
      device: item.run.device,
      tps64: item.metrics?.find((m: Metric) => m.name === 'tps_prompt_64')?.value_numeric ?? null as number | null,
      tps128: item.metrics?.find((m: Metric) => m.name === 'tps_prompt_128')?.value_numeric ?? null as number | null,
      tps256: item.metrics?.find((m: Metric) => m.name === 'tps_prompt_256')?.value_numeric ?? null as number | null,
    };
  });

  // Sort by TPS (descending)
  const sortedByTps = [...tpsData].sort((a, b) => b.tps - a.tps);
  const winner = sortedByTps[0];

  // Prompt-length sweep: tps_prompt_64, tps_prompt_128, tps_prompt_256
  // Shape: [{ promptLength: 64, [runLabel]: tps, ... }, ...]
  const promptSweepRuns = processedData.filter((item: any) =>
    PROMPT_LENGTHS.some((len) =>
      item.metrics?.some((m: Metric) => m.name === `tps_prompt_${len}`)
    )
  );
  const promptSweepData = PROMPT_LENGTHS.map((len) => {
    const point: Record<string, string | number> = { promptLength: len };
    promptSweepRuns.forEach((item: any) => {
      const label = buildRunLabel(item.run);
      const metric = item.metrics?.find((m: Metric) => m.name === `tps_prompt_${len}`);
      point[label] = metric?.value_numeric ?? 0;
    });
    return point;
  });
  const promptSweepLines = promptSweepRuns.map((item: any, idx: number) => {
    const label = buildRunLabel(item.run);
    return {
      key: label,
      name: label,
      color: SWEEP_PALETTE[idx % SWEEP_PALETTE.length],
    };
  });

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
              data={sortedByTps.map((item, idx) => ({
                name: item.modelId.replace('-GGUF', '').split('-').pop() || item.modelId,
                value: item.tps,
                color: SWEEP_PALETTE[idx % SWEEP_PALETTE.length],
              }))}
              dataKey="value"
              title="Tokens per Second (higher is better)"
              unit="tok/s"
              height={300}
              highlightBest
            />
          </Card>

          {/* Prompt-Length Sweep Chart */}
          <Card padding="lg" radius="md" withBorder mb="md">
            <Title order={4} mb="xs">
              TPS vs Prompt Length
            </Title>
            <Text size="sm" c="dimmed" mb="md">
              How token generation speed changes as prompt length increases (64 → 128 → 256 tokens)
            </Text>
            {promptSweepRuns.length === 0 ? (
              <Text c="dimmed" ta="center" py="xl" size="sm">
                No prompt-length sweep data available. Run the bench command with prompt length
                variants to populate this chart.
              </Text>
            ) : (
              <LineChart
                data={promptSweepData as any}
                dataKey="promptLength"
                nameKey="promptLength"
                multipleLines={promptSweepLines}
                unit="tok/s"
                height={320}
                yAxisLabel="Tokens / Second"
              />
            )}
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
                  <Table.Th ta="right">TPS@64</Table.Th>
                  <Table.Th ta="right">TPS@128</Table.Th>
                  <Table.Th ta="right">TPS@256</Table.Th>
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
                      <Text size="sm" c="dimmed">{item.tps64?.toFixed(2) ?? '-'}</Text>
                    </Table.Td>
                    <Table.Td ta="right">
                      <Text size="sm" c="dimmed">{item.tps128?.toFixed(2) ?? '-'}</Text>
                    </Table.Td>
                    <Table.Td ta="right">
                      <Text size="sm" c="dimmed">{item.tps256?.toFixed(2) ?? '-'}</Text>
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
