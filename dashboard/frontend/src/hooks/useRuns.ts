/**
 * React Query hooks for Runs
 */

import { useQuery, useMutation, useQueryClient, useQueryClient as getClient } from '@tanstack/react-query';
import { runsApi, type ListRunsParams } from '@/api/runs';
import type { Run, RunCreate, RunUpdate, RunStatus } from '@/types';
import { useNotificationStore } from '@/stores/notificationStore';

const QUERY_KEYS = {
  runs: {
    all: ['runs'] as const,
    lists: () => [...QUERY_KEYS.runs.all, 'list'] as const,
    list: (params: ListRunsParams) => [...QUERY_KEYS.runs.lists(), params] as const,
    details: () => [...QUERY_KEYS.runs.all, 'detail'] as const,
    detail: (id: string) => [...QUERY_KEYS.runs.details(), id] as const,
    metrics: (id: string) => [...QUERY_KEYS.runs.detail(id), 'metrics'] as const,
    recent: () => [...QUERY_KEYS.runs.all, 'recent'] as const,
    stats: () => [...QUERY_KEYS.runs.all, 'stats'] as const,
  },
};

export function useRuns(params?: ListRunsParams) {
  const query = useQuery({
    queryKey: QUERY_KEYS.runs.list(params || {}),
    queryFn: () => runsApi.listRuns(params),
  });

  return {
    ...query,
    runs: query.data?.data || [],
    meta: query.data?.meta,
  };
}

export function useRun(runId: string, includeMetrics = true, enabled = true) {
  const query = useQuery({
    queryKey: QUERY_KEYS.runs.detail(runId),
    queryFn: () => runsApi.getRun(runId, includeMetrics),
    enabled: !!runId && enabled,
  });

  return {
    ...query,
    run: query.data?.data || null,
  };
}

export function useRunMetrics(runId: string, enabled = true) {
  const query = useQuery({
    queryKey: QUERY_KEYS.runs.metrics(runId),
    queryFn: () => runsApi.getRunMetrics(runId),
    enabled: !!runId && enabled,
  });

  return {
    ...query,
    metrics: query.data?.data || [],
  };
}

export function useCreateRun() {
  const queryClient = useQueryClient();
  const addNotification = useNotificationStore((state) => state.addNotification);

  return useMutation({
    mutationFn: (data: RunCreate) => runsApi.createRun(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.runs.all });
      addNotification({
        type: 'success',
        title: 'Run created',
        message: 'The evaluation run was created successfully',
      });
    },
    onError: (error: Error) => {
      addNotification({
        type: 'error',
        title: 'Failed to create run',
        message: error.message,
      });
    },
  });
}

export function useUpdateRun(runId: string) {
  const queryClient = useQueryClient();
  const addNotification = useNotificationStore((state) => state.addNotification);

  return useMutation({
    mutationFn: (data: RunUpdate) => runsApi.updateRun(runId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.runs.detail(runId) });
      addNotification({
        type: 'success',
        title: 'Run updated',
        message: 'The run was updated successfully',
      });
    },
    onError: (error: Error) => {
      addNotification({
        type: 'error',
        title: 'Failed to update run',
        message: error.message,
      });
    },
  });
}

export function useUpdateRunStatus(runId: string) {
  const queryClient = useQueryClient();
  const addNotification = useNotificationStore((state) => state.addNotification);

  return useMutation({
    mutationFn: ({ status, message }: { status: RunStatus; message?: string }) =>
      runsApi.updateRunStatus(runId, status, message),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.runs.detail(runId) });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.runs.all });
    },
    onError: (error: Error) => {
      addNotification({
        type: 'error',
        title: 'Failed to update run status',
        message: error.message,
      });
    },
  });
}

export function useDeleteRun() {
  const queryClient = useQueryClient();
  const addNotification = useNotificationStore((state) => state.addNotification);

  return useMutation({
    mutationFn: (runId: string) => runsApi.deleteRun(runId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.runs.all });
      addNotification({
        type: 'success',
        title: 'Run deleted',
        message: 'The run was deleted successfully',
      });
    },
    onError: (error: Error) => {
      addNotification({
        type: 'error',
        title: 'Failed to delete run',
        message: error.message,
      });
    },
  });
}

export function useRecentRuns(limit = 10) {
  const query = useQuery({
    queryKey: QUERY_KEYS.runs.recent(),
    queryFn: () => runsApi.getRecentRuns(limit),
    refetchInterval: import.meta.env.VITE_POLLING_INTERVAL_SLOW
      ? parseInt(import.meta.env.VITE_POLLING_INTERVAL_SLOW, 10) * 1000
      : 15000, // Default 15 seconds
  });

  return {
    ...query,
    runs: query.data?.data || [],
  };
}

export function useRunStats() {
  const query = useQuery({
    queryKey: QUERY_KEYS.runs.stats(),
    queryFn: () => runsApi.getRunStats(),
    refetchInterval: import.meta.env.VITE_POLLING_INTERVAL_FAST
      ? parseInt(import.meta.env.VITE_POLLING_INTERVAL_FAST, 10) * 1000
      : 30000, // Default 30 seconds
  });

  return {
    ...query,
    stats: query.data?.data || null,
  };
}

export { QUERY_KEYS };
