/**
 * Radar Chart Component for Multi-Metric Comparison
 * Built with Recharts
 */

import {
  RadarChart as RechartsRadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Box, Text, Group, useMantineColorScheme } from '@mantine/core';

export interface RadarChartData {
  subject: string;
  A?: number;
  B?: number;
  fullMark: number;
  [key: string]: string | number;
}

interface RadarChartProps {
  data: RadarChartData[];
  dataKeys: { key: string; name: string; color: string }[];
  title?: string;
  unit?: string;
  height?: number;
  showGrid?: boolean;
  showLegend?: boolean;
}

export function RadarChart({
  data,
  dataKeys,
  title,
  unit,
  height = 400,
  showGrid = true,
  showLegend = true,
}: RadarChartProps) {
  const { colorScheme } = useMantineColorScheme();
  const textColor = colorScheme === 'dark' ? '#9ca3af' : '#6b7280';
  const gridColor = colorScheme === 'dark' ? '#374151' : '#e5e7eb';

  const formatTooltipValue = (value: number) =>
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
        <RechartsRadarChart
          cx="50%"
          cy="50%"
          outerRadius="80%"
          data={data}
          margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
        >
          {showGrid && <PolarGrid stroke={gridColor} />}
          <PolarAngleAxis
            dataKey="subject"
            tick={{ fill: textColor, fontSize: 12 }}
          />
          <PolarRadiusAxis
            stroke={textColor}
            tick={{ fill: textColor, fontSize: 10 }}
            tickFormatter={(value: number) => `${value}${unit ? ` ${unit}` : ''}`}
          />
          {dataKeys.map((dataKey) => (
            <Radar
              key={dataKey.key}
              name={dataKey.name}
              dataKey={dataKey.key}
              stroke={dataKey.color}
              fill={dataKey.color}
              fillOpacity={0.3}
              strokeWidth={2}
            />
          ))}
          {showLegend && <Legend />}
          <Tooltip
            formatter={formatTooltipValue}
            contentStyle={{
              backgroundColor: colorScheme === 'dark' ? '#1f2937' : '#ffffff',
              border: `1px solid ${gridColor}`,
              borderRadius: 8,
            }}
            labelStyle={{ color: textColor }}
          />
        </RechartsRadarChart>
      </ResponsiveContainer>
    </Box>
  );
}

export default RadarChart;
