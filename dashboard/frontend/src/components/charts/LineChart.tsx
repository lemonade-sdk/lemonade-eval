/**
 * Line Chart Component for Trends
 * Built with Recharts
 */

import {
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { Box, Text, Group, useMantineColorScheme } from '@mantine/core';

export interface LineChartData {
  name: string;
  value: number;
  [key: string]: string | number;
}

interface LineChartProps {
  data: LineChartData[];
  dataKey: string;
  nameKey?: string;
  title?: string;
  unit?: string;
  height?: number;
  showGrid?: boolean;
  showLegend?: boolean;
  yAxisLabel?: string;
  color?: string;
  multipleLines?: { key: string; name: string; color: string }[];
}

export function LineChart({
  data,
  dataKey,
  nameKey = 'name',
  title,
  unit,
  height = 300,
  showGrid = true,
  showLegend = true,
  yAxisLabel,
  color = 'var(--mantine-color-blue-6)',
  multipleLines,
}: LineChartProps) {
  const { colorScheme } = useMantineColorScheme();
  const textColor = colorScheme === 'dark' ? '#9ca3af' : '#6b7280';
  const gridColor = colorScheme === 'dark' ? '#374151' : '#e5e7eb';

  const formatTooltipValue = (value: number) =>
    `${value.toLocaleString()}${unit ? ` ${unit}` : ''}`;

  const formatYAxisTick = (value: number) =>
    `${value.toLocaleString()}${unit ? ` ${unit}` : ''}`;

  return (
    <Box>
      {title && (
        <Group mb="md">
          <Text fw={600} size="lg">
            {title}
          </Text>
        </Group>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <RechartsLineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 25 }}>
          {showGrid && <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />}
          <XAxis
            dataKey={nameKey}
            stroke={textColor}
            tick={{ fontSize: 12 }}
            angle={-45}
            textAnchor="end"
            height={60}
          />
          <YAxis
            stroke={textColor}
            tick={{ fontSize: 12 }}
            tickFormatter={formatYAxisTick}
            label={yAxisLabel ? { value: yAxisLabel, angle: -90, position: 'insideLeft', fill: textColor } : undefined}
          />
          <Tooltip
            formatter={formatTooltipValue}
            contentStyle={{
              backgroundColor: colorScheme === 'dark' ? '#1f2937' : '#ffffff',
              border: `1px solid ${gridColor}`,
              borderRadius: 8,
            }}
            labelStyle={{ color: textColor }}
          />
          {showLegend && <Legend />}
          {multipleLines ? (
            multipleLines.map((line) => (
              <Line
                key={line.key}
                type="monotone"
                dataKey={line.key}
                name={line.name}
                stroke={line.color}
                strokeWidth={2}
                dot={{ r: 4 }}
                activeDot={{ r: 6 }}
              />
            ))
          ) : (
            <Line
              type="monotone"
              dataKey={dataKey}
              stroke={color}
              strokeWidth={2}
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
            />
          )}
        </RechartsLineChart>
      </ResponsiveContainer>
    </Box>
  );
}

export default LineChart;
