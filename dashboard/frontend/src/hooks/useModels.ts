/**
 * React Query hooks for Models
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { modelsApi } from '@/api/models';
import type { Model, ModelCreate, ModelUpdate } from '@/types';
import type { ListRunsParams } from '@/api/runs';
import { useNotificationStore } from '@/stores/notificationStore';

const QUERY_KEYS = {
  models: {
    all: ['models'] as const,
    lists: () => [...QUERY_KEYS.models.all, 'list'] as const,
    list: (params: ListRunsParams) => [...QUERY_KEYS.models.lists(), params] as const,
    details: () => [...QUERY_KEYS.models.all, 'detail'] as const,
    detail: (id: string) => [...QUERY_KEYS.models.details(), id] as const,
    families: () => [...QUERY_KEYS.models.all, 'families'] as const,
    versions: (id: string) => [...QUERY_KEYS.models.detail(id), 'versions'] as const,
    runs: (id: string) => [...QUERY_KEYS.models.detail(id), 'runs'] as const,
  },
};

export function useModels(params?: {
  page?: number;
  per_page?: number;
  search?: string | null;
  family?: string | null;
  model_type?: string | null;
}) {
  const query = useQuery({
    queryKey: QUERY_KEYS.models.list(params || {}),
    queryFn: () => modelsApi.listModels(params),
  });

  return {
    ...query,
    models: query.data?.data || [],
    meta: query.data?.meta,
  };
}

export function useModel(modelId: string, enabled = true) {
  const query = useQuery({
    queryKey: QUERY_KEYS.models.detail(modelId),
    queryFn: () => modelsApi.getModel(modelId),
    enabled: !!modelId && enabled,
  });

  return {
    ...query,
    model: query.data?.data || null,
  };
}

export function useCreateModel() {
  const queryClient = useQueryClient();
  const addNotification = useNotificationStore((state) => state.addNotification);

  return useMutation({
    mutationFn: (data: ModelCreate) => modelsApi.createModel(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.models.all });
      addNotification({
        type: 'success',
        title: 'Model created',
        message: 'The model was created successfully',
      });
    },
    onError: (error: Error) => {
      addNotification({
        type: 'error',
        title: 'Failed to create model',
        message: error.message,
      });
    },
  });
}

export function useUpdateModel(modelId: string) {
  const queryClient = useQueryClient();
  const addNotification = useNotificationStore((state) => state.addNotification);

  return useMutation({
    mutationFn: (data: ModelUpdate) => modelsApi.updateModel(modelId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.models.detail(modelId) });
      addNotification({
        type: 'success',
        title: 'Model updated',
        message: 'The model was updated successfully',
      });
    },
    onError: (error: Error) => {
      addNotification({
        type: 'error',
        title: 'Failed to update model',
        message: error.message,
      });
    },
  });
}

export function useDeleteModel() {
  const queryClient = useQueryClient();
  const addNotification = useNotificationStore((state) => state.addNotification);

  return useMutation({
    mutationFn: (modelId: string) => modelsApi.deleteModel(modelId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.models.all });
      addNotification({
        type: 'success',
        title: 'Model deleted',
        message: 'The model was deleted successfully',
      });
    },
    onError: (error: Error) => {
      addNotification({
        type: 'error',
        title: 'Failed to delete model',
        message: error.message,
      });
    },
  });
}

export function useModelFamilies() {
  const query = useQuery({
    queryKey: QUERY_KEYS.models.families(),
    queryFn: () => modelsApi.listFamilies(),
  });

  return {
    ...query,
    families: query.data?.data || [],
  };
}

export function useModelVersions(modelId: string) {
  const query = useQuery({
    queryKey: QUERY_KEYS.models.versions(modelId),
    queryFn: () => modelsApi.getModelVersions(modelId),
    enabled: !!modelId,
  });

  return {
    ...query,
    versions: query.data?.data || [],
  };
}

export function useModelRuns(modelId: string, limit?: number) {
  const query = useQuery({
    queryKey: QUERY_KEYS.models.runs(modelId),
    queryFn: () => modelsApi.getModelRuns(modelId, limit),
    enabled: !!modelId,
  });

  return {
    ...query,
    runs: query.data?.data || [],
  };
}

export { QUERY_KEYS };
