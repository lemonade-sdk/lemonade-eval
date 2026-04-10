/**
 * Bar Chart Component for Comparisons
 * Built with Recharts
 */

import {
  BarChart as RechartsBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { Box, Text, Group, useMantineColorScheme } from '@mantine/core';

export interface BarChartData {
  name: string;
  value: number;
  [key: string]: string | number;
}

interface BarChartProps {
  data: BarChartData[];
  dataKey: string;
  nameKey?: string;
  title?: string;
  unit?: string;
  height?: number;
  layout?: 'horizontal' | 'vertical';
  showGrid?: boolean;
  showLegend?: boolean;
  showValue?: boolean;
  color?: string;
  colors?: string[];
  highlightBest?: boolean;
}

export function BarChart({
  data,
  dataKey,
  nameKey = 'name',
  title,
  unit,
  height = 300,
  layout = 'horizontal',
  showGrid = true,
  showLegend = true,
  showValue = false,
  color = 'var(--mantine-color-blue-6)',
  colors,
  highlightBest = false,
}: BarChartProps) {
  const { colorScheme } = useMantineColorScheme();
  const textColor = colorScheme === 'dark' ? '#9ca3af' : '#6b7280';
  const gridColor = colorScheme === 'dark' ? '#374151' : '#e5e7eb';

  // Find best value for highlighting (highest for performance metrics)
  const maxValue = highlightBest ? Math.max(...data.map((d) => Number(d[dataKey]))) : null;

  const formatTooltipValue = (value: number) =>
    `${value.toLocaleString()}${unit ? ` ${unit}` : ''}`;

  const getBarColor = (value: number, index: number) => {
    if (colors && colors.length > 0) {
      return colors[index % colors.length];
    }
    if (highlightBest && value === maxValue) {
      return 'var(--mantine-color-green-6)';
    }
    return color;
  };

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
        <RechartsBarChart
          data={data}
          layout={layout}
          margin={{ top: 5, right: 30, left: 20, bottom: 25 }}
          barSize={layout === 'horizontal' ? 40 : 24}
        >
          {showGrid && <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />}
          <XAxis
            type={layout === 'horizontal' ? 'category' : 'number'}
            dataKey={layout === 'horizontal' ? nameKey : undefined}
            stroke={textColor}
            tick={{ fontSize: 12 }}
            angle={layout === 'horizontal' ? -45 : 0}
            textAnchor={layout === 'horizontal' ? 'end' : 'middle'}
            height={layout === 'horizontal' ? 60 : undefined}
            tickFormatter={layout === 'vertical' ? (v: number) => `${v}${unit ? ` ${unit}` : ''}` : undefined}
          />
          <YAxis
            type={layout === 'horizontal' ? 'number' : 'category'}
            dataKey={layout === 'vertical' ? nameKey : undefined}
            stroke={textColor}
            tick={{ fontSize: 12 }}
            width={layout === 'vertical' ? 150 : undefined}
            tickFormatter={layout === 'horizontal' ? (v: number) => `${v}${unit ? ` ${unit}` : ''}` : undefined}
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
          <Bar dataKey={dataKey} radius={[4, 4, 0, 0]}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={getBarColor(Number(entry[dataKey]), index)}
              />
            ))}
          </Bar>
        </RechartsBarChart>
      </ResponsiveContainer>
    </Box>
  );
}

export default BarChart;
