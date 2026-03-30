/**
 * React Query hooks for Metrics
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { metricsApi, type ListMetricsParams } from '@/api/metrics';
import type { Metric, MetricCreate } from '@/types';
import { useNotificationStore } from '@/stores/notificationStore';

const QUERY_KEYS = {
  metrics: {
    all: ['metrics'] as const,
    lists: () => [...QUERY_KEYS.metrics.all, 'list'] as const,
    list: (params: ListMetricsParams) => [...QUERY_KEYS.metrics.lists(), params] as const,
    detail: (id: string) => [...QUERY_KEYS.metrics.all, 'detail', id] as const,
    aggregate: (params?: Record<string, string | null>) => [...QUERY_KEYS.metrics.all, 'aggregate', params] as const,
    trends: (modelId: string, metricName: string) => [...QUERY_KEYS.metrics.all, 'trends', modelId, metricName] as const,
    compare: (runIds: string[]) => [...QUERY_KEYS.metrics.all, 'compare', runIds] as const,
    performance: (runId: string) => [...QUERY_KEYS.metrics.all, 'performance', runId] as const,
  },
};

export function useMetrics(params?: ListMetricsParams) {
  const query = useQuery({
    queryKey: QUERY_KEYS.metrics.list(params || {}),
    queryFn: () => metricsApi.listMetrics(params),
  });

  return {
    ...query,
    metrics: query.data?.data || [],
    meta: query.data?.meta,
  };
}

export function useMetric(metricId: string, enabled = true) {
  const query = useQuery({
    queryKey: QUERY_KEYS.metrics.detail(metricId),
    queryFn: () => metricsApi.getMetric(metricId),
    enabled: !!metricId && enabled,
  });

  return {
    ...query,
    metric: query.data?.data || null,
  };
}

export function useCreateMetric() {
  const queryClient = useQueryClient();
  const addNotification = useNotificationStore((state) => state.addNotification);

  return useMutation({
    mutationFn: (data: MetricCreate) => metricsApi.createMetric(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.metrics.all });
      addNotification({
        type: 'success',
        title: 'Metric created',
        message: 'The metric was created successfully',
      });
    },
    onError: (error: Error) => {
      addNotification({
        type: 'error',
        title: 'Failed to create metric',
        message: error.message,
      });
    },
  });
}

export function useCreateMetricsBulk() {
  const queryClient = useQueryClient();
  const addNotification = useNotificationStore((state) => state.addNotification);

  return useMutation({
    mutationFn: (metrics: MetricCreate[]) => metricsApi.createMetricsBulk(metrics),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.metrics.all });
      addNotification({
        type: 'success',
        title: 'Metrics created',
        message: `${metrics.length} metrics were created successfully`,
      });
    },
    onError: (error: Error) => {
      addNotification({
        type: 'error',
        title: 'Failed to create metrics',
        message: error.message,
      });
    },
  });
}

export function useDeleteMetric() {
  const queryClient = useQueryClient();
  const addNotification = useNotificationStore((state) => state.addNotification);

  return useMutation({
    mutationFn: (metricId: string) => metricsApi.deleteMetric(metricId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.metrics.all });
      addNotification({
        type: 'success',
        title: 'Metric deleted',
        message: 'The metric was deleted successfully',
      });
    },
    onError: (error: Error) => {
      addNotification({
        type: 'error',
        title: 'Failed to delete metric',
        message: error.message,
      });
    },
  });
}

export function useAggregateMetrics(params?: {
  model_id?: string | null;
  run_type?: string | null;
  category?: string | null;
  metric_name?: string | null;
}) {
  const query = useQuery({
    queryKey: QUERY_KEYS.metrics.aggregate(params || {}),
    queryFn: () => metricsApi.getAggregateMetrics(params),
  });

  return {
    ...query,
    aggregates: query.data?.data || [],
  };
}

export function useMetricTrends(modelId: string, metricName: string, limit = 100) {
  const query = useQuery({
    queryKey: QUERY_KEYS.metrics.trends(modelId, metricName),
    queryFn: () => metricsApi.getMetricTrends({ modelId, metricName, limit }),
    enabled: !!modelId && !!metricName,
  });

  return {
    ...query,
    trends: query.data?.data || [],
  };
}

export function useCompareMetrics(runIds: string[], categories?: string[]) {
  const query = useQuery({
    queryKey: QUERY_KEYS.metrics.compare(runIds),
    queryFn: () => metricsApi.compareMetrics(runIds, categories),
    enabled: runIds.length > 0,
  });

  return {
    ...query,
    comparison: query.data?.data || null,
  };
}

export function usePerformanceMetrics(runId: string, enabled = true) {
  const query = useQuery({
    queryKey: QUERY_KEYS.metrics.performance(runId),
    queryFn: () => metricsApi.getPerformanceMetrics(runId),
    enabled: !!runId && enabled,
  });

  return {
    ...query,
    performance: query.data?.data || null,
  };
}

export { QUERY_KEYS };
