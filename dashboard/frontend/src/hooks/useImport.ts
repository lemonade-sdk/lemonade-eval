/**
 * React Query hooks for Import operations
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { importApi } from '@/api/import';
import type { ImportRequest, ImportJobStatus } from '@/types';
import { useNotificationStore } from '@/stores/notificationStore';

const QUERY_KEYS = {
  import: {
    all: ['import'] as const,
    jobs: () => [...QUERY_KEYS.import.all, 'jobs'] as const,
    jobStatus: (jobId: string) => [...QUERY_KEYS.import.jobs(), jobId] as const,
  },
};

export function useImportYaml() {
  const queryClient = useQueryClient();
  const addNotification = useNotificationStore((state) => state.addNotification);

  return useMutation({
    mutationFn: (request: ImportRequest) => importApi.importYaml(request),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.import.all });
      return data;
    },
    onError: (error: Error) => {
      addNotification({
        type: 'error',
        title: 'Failed to start import',
        message: error.message,
      });
    },
  });
}

export function useImportStatus(jobId: string, enabled = true) {
  const query = useQuery({
    queryKey: QUERY_KEYS.import.jobStatus(jobId),
    queryFn: () => importApi.getImportStatus(jobId),
    enabled: !!jobId && enabled,
    refetchInterval: (query) => {
      const status = query.state.data?.data?.status;
      // Poll more frequently for active jobs
      if (status === 'pending' || status === 'running') {
        return import.meta.env.VITE_POLLING_INTERVAL_IMPORT
          ? parseInt(import.meta.env.VITE_POLLING_INTERVAL_IMPORT, 10) * 1000
          : 2000; // Default 2 seconds for import status
      }
      return false; // Stop polling for completed/failed jobs
    },
  });

  return {
    ...query,
    jobStatus: query.data?.data || null,
    isCompleted: query.data?.data?.status === 'completed',
    isFailed: query.data?.data?.status === 'failed',
    isRunning: query.data?.data?.status === 'running',
  };
}

export function useScanCacheDirectory() {
  const addNotification = useNotificationStore((state) => state.addNotification);

  return useMutation({
    mutationFn: (cacheDir: string) => importApi.scanCacheDirectory(cacheDir),
    onError: (error: Error) => {
      addNotification({
        type: 'error',
        title: 'Failed to scan directory',
        message: error.message,
      });
    },
  });
}

export function useImportJobs() {
  const query = useQuery({
    queryKey: QUERY_KEYS.import.jobs(),
    queryFn: () => importApi.listImportJobs(),
  });

  return {
    ...query,
    jobs: query.data?.data || [],
  };
}

export { QUERY_KEYS };
