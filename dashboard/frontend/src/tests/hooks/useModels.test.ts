/**
 * Tests for useModels hook
 */

import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

import { useModels, useModel, useCreateModel, useUpdateModel, useDeleteModel, useModelFamilies, QUERY_KEYS } from '../useModels';
import type { Model, ModelCreate, ModelUpdate } from '@/types';

// Mock data
const mockModels = {
  success: true,
  data: [
    {
      id: 'model-1',
      name: 'Llama-2b',
      checkpoint: 'meta/llama-2b',
      model_type: 'llm' as const,
      family: 'Llama',
      parameters: 2000000000,
      created_at: '2024-01-01T00:00:00Z',
    },
    {
      id: 'model-2',
      name: 'Qwen-1b',
      checkpoint: 'qwen-1b',
      model_type: 'llm' as const,
      family: 'Qwen',
      parameters: 1000000000,
      created_at: '2024-01-02T00:00:00Z',
    },
  ],
  meta: {
    page: 1,
    per_page: 20,
    total: 2,
    total_pages: 1,
  },
};

const mockModel = {
  success: true,
  data: {
    id: 'model-1',
    name: 'Llama-2b',
    checkpoint: 'meta/llama-2b',
    model_type: 'llm' as const,
    family: 'Llama',
    parameters: 2000000000,
    created_at: '2024-01-01T00:00:00Z',
  },
};

const mockFamilies = {
  success: true,
  data: ['Llama', 'Qwen', 'Phi'],
};

// MSW Server setup
const server = setupServer();

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => {
  server.resetHandlers();
  vi.clearAllMocks();
});
afterAll(() => server.close());

// Test utilities
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: 0,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

describe('useModels', () => {
  it('should fetch models successfully', async () => {
    server.use(
      http.get('/api/v1/models', () => HttpResponse.json(mockModels))
    );

    const { result } = renderHook(() => useModels(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.models).toHaveLength(2);
    expect(result.current.meta?.total).toBe(2);
    expect(result.current.isSuccess).toBe(true);
  });

  it('should fetch models with params', async () => {
    server.use(
      http.get('/api/v1/models', () => HttpResponse.json(mockModels))
    );

    const params = {
      page: 1,
      per_page: 10,
      search: 'Llama',
      family: 'Llama',
      model_type: 'llm',
    };

    const { result } = renderHook(() => useModels(params), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.models).toHaveLength(2);
  });

  it('should handle fetch error', async () => {
    server.use(
      http.get('/api/v1/models', () =>
        HttpResponse.json({ detail: 'Failed to fetch' }, { status: 500 })
      )
    );

    const { result } = renderHook(() => useModels(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.isError).toBe(true);
  });

  it('should return empty array when no models', async () => {
    server.use(
      http.get('/api/v1/models', () =>
        HttpResponse.json({ success: true, data: [], meta: { page: 1, per_page: 20, total: 0, total_pages: 0 } })
      )
    );

    const { result } = renderHook(() => useModels(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.models).toHaveLength(0);
    expect(result.current.meta?.total).toBe(0);
  });
});

describe('useModel', () => {
  it('should fetch single model successfully', async () => {
    server.use(
      http.get('/api/v1/models/model-1', () => HttpResponse.json(mockModel))
    );

    const { result } = renderHook(() => useModel('model-1'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.model).toBeTruthy();
    expect(result.current.model?.id).toBe('model-1');
    expect(result.current.model?.name).toBe('Llama-2b');
  });

  it('should not fetch when modelId is empty', () => {
    const { result } = renderHook(() => useModel(''), {
      wrapper: createWrapper(),
    });

    expect(result.current.model).toBeNull();
    expect(result.current.isLoading).toBe(false);
  });

  it('should respect enabled option', () => {
    const { result } = renderHook(() => useModel('model-1', false), {
      wrapper: createWrapper(),
    });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.model).toBeNull();
  });

  it('should handle fetch error', async () => {
    server.use(
      http.get('/api/v1/models/model-1', () =>
        HttpResponse.json({ detail: 'Model not found' }, { status: 404 })
      )
    );

    const { result } = renderHook(() => useModel('model-1'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.isError).toBe(true);
  });
});

describe('useModelFamilies', () => {
  it('should fetch families successfully', async () => {
    server.use(
      http.get('/api/v1/models/families/list', () => HttpResponse.json(mockFamilies))
    );

    const { result } = renderHook(() => useModelFamilies(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.families).toHaveLength(3);
    expect(result.current.families).toEqual(['Llama', 'Qwen', 'Phi']);
  });

  it('should handle empty families', async () => {
    server.use(
      http.get('/api/v1/models/families/list', () =>
        HttpResponse.json({ success: true, data: [] })
      )
    );

    const { result } = renderHook(() => useModelFamilies(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.families).toHaveLength(0);
  });
});

describe('useCreateModel', () => {
  it('should create model successfully', async () => {
    const newModel: ModelCreate = {
      name: 'New Model',
      checkpoint: 'test/new-model',
    };

    server.use(
      http.post('/api/v1/models', async () =>
        HttpResponse.json({
          success: true,
          data: { id: 'new-id', ...newModel }
        }, { status: 201 })
      )
    );

    const { result } = renderHook(() => useCreateModel(), {
      wrapper: createWrapper(),
    });

    result.current.mutate(newModel);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.success).toBe(true);
  });

  it('should handle create error', async () => {
    server.use(
      http.post('/api/v1/models', () =>
        HttpResponse.json({ detail: 'Checkpoint already exists' }, { status: 400 })
      )
    );

    const { result } = renderHook(() => useCreateModel(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({ name: 'Duplicate', checkpoint: 'existing/checkpoint' });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useUpdateModel', () => {
  it('should update model successfully', async () => {
    const updateData: ModelUpdate = { name: 'Updated Name' };

    server.use(
      http.put('/api/v1/models/model-1', async () =>
        HttpResponse.json({
          success: true,
          data: { id: 'model-1', ...updateData }
        })
      )
    );

    const { result } = renderHook(() => useUpdateModel('model-1'), {
      wrapper: createWrapper(),
    });

    result.current.mutate(updateData);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
  });

  it('should handle update error', async () => {
    server.use(
      http.put('/api/v1/models/model-1', () =>
        HttpResponse.json({ detail: 'Model not found' }, { status: 404 })
      )
    );

    const { result } = renderHook(() => useUpdateModel('model-1'), {
      wrapper: createWrapper(),
    });

    result.current.mutate({ name: 'Updated' });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useDeleteModel', () => {
  it('should delete model successfully', async () => {
    server.use(
      http.delete('/api/v1/models/model-1', () =>
        HttpResponse.json({ success: true, data: { message: 'Deleted' } })
      )
    );

    const { result } = renderHook(() => useDeleteModel(), {
      wrapper: createWrapper(),
    });

    result.current.mutate('model-1');

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
  });

  it('should handle delete error', async () => {
    server.use(
      http.delete('/api/v1/models/model-1', () =>
        HttpResponse.json({ detail: 'Model not found' }, { status: 404 })
      )
    );

    const { result } = renderHook(() => useDeleteModel(), {
      wrapper: createWrapper(),
    });

    result.current.mutate('model-1');

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('QUERY_KEYS', () => {
  it('should have correct query key structure', () => {
    expect(QUERY_KEYS.models.all).toEqual(['models']);
    expect(QUERY_KEYS.models.lists()).toEqual(['models', 'list']);
    expect(QUERY_KEYS.models.detail('model-1')).toEqual(['models', 'detail', 'model-1']);
    expect(QUERY_KEYS.models.families()).toEqual(['models', 'families']);
    expect(QUERY_KEYS.models.versions('model-1')).toEqual(['models', 'detail', 'model-1', 'versions']);
    expect(QUERY_KEYS.models.runs('model-1')).toEqual(['models', 'detail', 'model-1', 'runs']);
  });
});
