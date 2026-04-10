/**
 * Tests for useRuns hook
 */

import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';

import {
  useRuns,
  useRun,
  useRunMetrics,
  useCreateRun,
  useUpdateRun,
  useUpdateRunStatus,
  useDeleteRun,
  useRecentRuns,
  useRunStats,
  QUERY_KEYS,
} from '../useRuns';
import type { RunCreate, RunUpdate, RunStatus } from '@/types';

// Mock data
const mockRuns = {
  success: true,
  data: [
    {
      id: 'run-1',
      model_id: 'model-1',
      build_name: 'test_build_001',
      run_type: 'benchmark',
      status: 'completed',
      device: 'gpu',
      backend: 'llamacpp',
      created_at: '2024-01-01T00:00:00Z',
    },
    {
      id: 'run-2',
      model_id: 'model-1',
      build_name: 'test_build_002',
      run_type: 'accuracy-mmlu',
      status: 'running',
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

const mockRun = {
  success: true,
  data: {
    id: 'run-1',
    model_id: 'model-1',
    build_name: 'test_build_001',
    run_type: 'benchmark',
    status: 'completed',
    device: 'gpu',
    backend: 'llamacpp',
    created_at: '2024-01-01T00:00:00Z',
  },
};

const mockRunWithMetrics = {
  success: true,
  data: {
    ...mockRun.data,
    metrics: [
      {
        id: 'metric-1',
        run_id: 'run-1',
        category: 'performance',
        name: 'seconds_to_first_token',
        value_numeric: 0.025,
        unit: 'seconds',
      },
    ],
  },
};

const mockMetrics = {
  success: true,
  data: [
    {
      id: 'metric-1',
      run_id: 'run-1',
      category: 'performance',
      name: 'seconds_to_first_token',
      value_numeric: 0.025,
      unit: 'seconds',
    },
    {
      id: 'metric-2',
      run_id: 'run-1',
      category: 'performance',
      name: 'token_generation_tokens_per_second',
      value_numeric: 45.5,
      unit: 'tokens/s',
    },
  ],
};

const mockRecentRuns = {
  success: true,
  data: mockRuns.data.slice(0, 1),
};

const mockRunStats = {
  success: true,
  data: {
    total_runs: 10,
    by_status: {
      completed: 7,
      running: 2,
      failed: 1,
    },
    by_type: {
      benchmark: 8,
      'accuracy-mmlu': 2,
    },
  },
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

describe('useRuns', () => {
  it('should fetch runs successfully', async () => {
    server.use(
      http.get('/api/v1/runs', () => HttpResponse.json(mockRuns))
    );

    const { result } = renderHook(() => useRuns(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.runs).toHaveLength(2);
    expect(result.current.meta?.total).toBe(2);
    expect(result.current.isSuccess).toBe(true);
  });

  it('should fetch runs with params', async () => {
    server.use(
      http.get('/api/v1/runs', () => HttpResponse.json(mockRuns))
    );

    const params = {
      page: 1,
      per_page: 10,
      model_id: 'model-1',
      status: 'completed',
      run_type: 'benchmark',
      device: 'gpu',
      backend: 'llamacpp',
    };

    const { result } = renderHook(() => useRuns(params), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.runs).toHaveLength(2);
  });

  it('should handle fetch error', async () => {
    server.use(
      http.get('/api/v1/runs', () =>
        HttpResponse.json({ detail: 'Failed to fetch' }, { status: 500 })
      )
    );

    const { result } = renderHook(() => useRuns(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.isError).toBe(true);
  });

  it('should return empty array when no runs', async () => {
    server.use(
      http.get('/api/v1/runs', () =>
        HttpResponse.json({ success: true, data: [], meta: { page: 1, per_page: 20, total: 0, total_pages: 0 } })
      )
    );

    const { result } = renderHook(() => useRuns(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.runs).toHaveLength(0);
    expect(result.current.meta?.total).toBe(0);
  });
});

describe('useRun', () => {
  it('should fetch single run successfully', async () => {
    server.use(
      http.get('/api/v1/runs/run-1', () => HttpResponse.json(mockRun))
    );

    const { result } = renderHook(() => useRun('run-1'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.run).toBeTruthy();
    expect(result.current.run?.id).toBe('run-1');
    expect(result.current.run?.build_name).toBe('test_build_001');
  });

  it('should fetch run with metrics', async () => {
    server.use(
      http.get('/api/v1/runs/run-1', () => HttpResponse.json(mockRunWithMetrics))
    );

    const { result } = renderHook(() => useRun('run-1', true), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.run?.metrics).toHaveLength(1);
  });

  it('should not fetch when runId is empty', () => {
    const { result } = renderHook(() => useRun(''), {
      wrapper: createWrapper(),
    });

    expect(result.current.run).toBeNull();
    expect(result.current.isLoading).toBe(false);
  });

  it('should respect enabled option', () => {
    const { result } = renderHook(() => useRun('run-1', true, false), {
      wrapper: createWrapper(),
    });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.run).toBeNull();
  });

  it('should handle fetch error', async () => {
    server.use(
      http.get('/api/v1/runs/run-1', () =>
        HttpResponse.json({ detail: 'Run not found' }, { status: 404 })
      )
    );

    const { result } = renderHook(() => useRun('run-1'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.isError).toBe(true);
  });
});

describe('useRunMetrics', () => {
  it('should fetch run metrics successfully', async () => {
    server.use(
      http.get('/api/v1/runs/run-1/metrics', () => HttpResponse.json(mockMetrics))
    );

    const { result } = renderHook(() => useRunMetrics('run-1'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.metrics).toHaveLength(2);
    expect(result.current.isSuccess).toBe(true);
  });

  it('should return empty array when no metrics', async () => {
    server.use(
      http.get('/api/v1/runs/run-1/metrics', () =>
        HttpResponse.json({ success: true, data: [] })
      )
    );

    const { result } = renderHook(() => useRunMetrics('run-1'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.metrics).toHaveLength(0);
  });

  it('should not fetch when runId is empty', () => {
    const { result } = renderHook(() => useRunMetrics(''), {
      wrapper: createWrapper(),
    });

    expect(result.current.metrics).toHaveLength(0);
  });
});

describe('useCreateRun', () => {
  it('should create run successfully', async () => {
    const newRun: RunCreate = {
      model_id: 'model-1',
      build_name: 'new_build',
      run_type: 'benchmark',
    };

    server.use(
      http.post('/api/v1/runs', async () =>
        HttpResponse.json({
          success: true,
          data: { id: 'new-run', ...newRun }
        }, { status: 201 })
      )
    );

    const { result } = renderHook(() => useCreateRun(), {
      wrapper: createWrapper(),
    });

    result.current.mutate(newRun);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.success).toBe(true);
  });

  it('should handle create error', async () => {
    server.use(
      http.post('/api/v1/runs', () =>
        HttpResponse.json({ detail: 'Invalid model_id' }, { status: 400 })
      )
    );

    const { result } = renderHook(() => useCreateRun(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({
      model_id: 'invalid',
      build_name: 'test',
      run_type: 'benchmark',
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useUpdateRun', () => {
  it('should update run successfully', async () => {
    const updateData: RunUpdate = { status: 'completed' };

    server.use(
      http.put('/api/v1/runs/run-1', async () =>
        HttpResponse.json({
          success: true,
          data: { id: 'run-1', ...updateData }
        })
      )
    );

    const { result } = renderHook(() => useUpdateRun('run-1'), {
      wrapper: createWrapper(),
    });

    result.current.mutate(updateData);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
  });

  it('should handle update error', async () => {
    server.use(
      http.put('/api/v1/runs/run-1', () =>
        HttpResponse.json({ detail: 'Run not found' }, { status: 404 })
      )
    );

    const { result } = renderHook(() => useUpdateRun('run-1'), {
      wrapper: createWrapper(),
    });

    result.current.mutate({ status: 'completed' });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useUpdateRunStatus', () => {
  it('should update run status successfully', async () => {
    server.use(
      http.post('/api/v1/runs/run-1/status', () =>
        HttpResponse.json({
          success: true,
          data: { id: 'run-1', status: 'completed' }
        })
      )
    );

    const { result } = renderHook(() => useUpdateRunStatus('run-1'), {
      wrapper: createWrapper(),
    });

    result.current.mutate({ status: 'completed' });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
  });

  it('should update run status with message', async () => {
    server.use(
      http.post('/api/v1/runs/run-1/status', () =>
        HttpResponse.json({
          success: true,
          data: {
            id: 'run-1',
            status: 'failed',
            status_message: 'Out of memory',
          }
        })
      )
    );

    const { result } = renderHook(() => useUpdateRunStatus('run-1'), {
      wrapper: createWrapper(),
    });

    result.current.mutate({ status: 'failed', message: 'Out of memory' });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
  });

  it('should handle invalid status', async () => {
    server.use(
      http.post('/api/v1/runs/run-1/status', () =>
        HttpResponse.json({ detail: 'Invalid status' }, { status: 400 })
      )
    );

    const { result } = renderHook(() => useUpdateRunStatus('run-1'), {
      wrapper: createWrapper(),
    });

    result.current.mutate({ status: 'invalid_status' as RunStatus });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useDeleteRun', () => {
  it('should delete run successfully', async () => {
    server.use(
      http.delete('/api/v1/runs/run-1', () =>
        HttpResponse.json({ success: true, data: { message: 'Deleted' } })
      )
    );

    const { result } = renderHook(() => useDeleteRun(), {
      wrapper: createWrapper(),
    });

    result.current.mutate('run-1');

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
  });

  it('should handle delete error', async () => {
    server.use(
      http.delete('/api/v1/runs/run-1', () =>
        HttpResponse.json({ detail: 'Run not found' }, { status: 404 })
      )
    );

    const { result } = renderHook(() => useDeleteRun(), {
      wrapper: createWrapper(),
    });

    result.current.mutate('run-1');

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useRecentRuns', () => {
  it('should fetch recent runs successfully', async () => {
    server.use(
      http.get('/api/v1/runs/recent/list', () => HttpResponse.json(mockRecentRuns))
    );

    const { result } = renderHook(() => useRecentRuns(10), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.runs).toHaveLength(1);
  });

  it('should use default limit of 10', async () => {
    server.use(
      http.get('/api/v1/runs/recent/list', () => HttpResponse.json(mockRecentRuns))
    );

    const { result } = renderHook(() => useRecentRuns(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.isSuccess).toBe(true);
  });
});

describe('useRunStats', () => {
  it('should fetch run stats successfully', async () => {
    server.use(
      http.get('/api/v1/runs/stats', () => HttpResponse.json(mockRunStats))
    );

    const { result } = renderHook(() => useRunStats(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.stats).toBeTruthy();
    expect(result.current.stats?.total_runs).toBe(10);
    expect(result.current.stats?.by_status.completed).toBe(7);
  });

  it('should handle empty stats', async () => {
    server.use(
      http.get('/api/v1/runs/stats', () =>
        HttpResponse.json({
          success: true,
          data: { total_runs: 0, by_status: {}, by_type: {} }
        })
      )
    );

    const { result } = renderHook(() => useRunStats(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.stats?.total_runs).toBe(0);
  });
});

describe('QUERY_KEYS', () => {
  it('should have correct query key structure', () => {
    expect(QUERY_KEYS.runs.all).toEqual(['runs']);
    expect(QUERY_KEYS.runs.lists()).toEqual(['runs', 'list']);
    expect(QUERY_KEYS.runs.list({})).toEqual(['runs', 'list', {}]);
    expect(QUERY_KEYS.runs.detail('run-1')).toEqual(['runs', 'detail', 'run-1']);
    expect(QUERY_KEYS.runs.metrics('run-1')).toEqual(['runs', 'detail', 'run-1', 'metrics']);
    expect(QUERY_KEYS.runs.recent()).toEqual(['runs', 'recent']);
    expect(QUERY_KEYS.runs.stats()).toEqual(['runs', 'stats']);
  });
});
