/**
 * API methods for Runs
 */

import apiClient from './client';
import type { Run, RunCreate, RunUpdate, RunListResponse, RunDetail, Metric, APIResponse } from '@/types';

const BASE_PATH = '/api/v1/runs';

export interface ListRunsParams {
  page?: number;
  per_page?: number;
  model_id?: string | null;
  status?: string | null;
  run_type?: string | null;
  device?: string | null;
  backend?: string | null;
}

export const runsApi = {
  /**
   * List runs with pagination and filtering
   */
  async listRuns(params?: ListRunsParams): Promise<RunListResponse> {
    const { data } = await apiClient.get<RunListResponse>(BASE_PATH, { params });
    return data;
  },

  /**
   * Get a specific run by ID
   */
  async getRun(runId: string, includeMetrics?: boolean): Promise<APIResponse<RunDetail>> {
    const { data } = await apiClient.get<APIResponse<RunDetail>>(`${BASE_PATH}/${runId}`, {
      params: { include_metrics: includeMetrics },
    });
    return data;
  },

  /**
   * Create a new run
   */
  async createRun(runData: RunCreate): Promise<APIResponse<Run>> {
    const { data } = await apiClient.post<APIResponse<Run>>(BASE_PATH, runData);
    return data;
  },

  /**
   * Update an existing run
   */
  async updateRun(runId: string, runData: RunUpdate): Promise<APIResponse<Run>> {
    const { data } = await apiClient.put<APIResponse<Run>>(`${BASE_PATH}/${runId}`, runData);
    return data;
  },

  /**
   * Update run status
   */
  async updateRunStatus(
    runId: string,
    status: string,
    message?: string | null
  ): Promise<APIResponse<Run>> {
    const { data } = await apiClient.post<APIResponse<Run>>(
      `${BASE_PATH}/${runId}/status`,
      {},
      { params: { status, message } }
    );
    return data;
  },

  /**
   * Delete a run
   */
  async deleteRun(runId: string): Promise<APIResponse<{ message: string }>> {
    const { data } = await apiClient.delete<APIResponse<{ message: string }>>(`${BASE_PATH}/${runId}`);
    return data;
  },

  /**
   * Get metrics for a run
   */
  async getRunMetrics(runId: string): Promise<APIResponse<Metric[]>> {
    const { data } = await apiClient.get<APIResponse<Metric[]>>(`${BASE_PATH}/${runId}/metrics`);
    return data;
  },

  /**
   * Get recent runs
   */
  async getRecentRuns(limit?: number): Promise<APIResponse<Run[]>> {
    const { data } = await apiClient.get<APIResponse<Run[]>>(`${BASE_PATH}/recent/list`, {
      params: { limit },
    });
    return data;
  },

  /**
   * Get run statistics
   */
  async getRunStats(): Promise<APIResponse<{
    total_runs: number;
    by_status: Record<string, number>;
    by_type: Record<string, number>;
  }>> {
    const { data } = await apiClient.get<APIResponse<{
      total_runs: number;
      by_status: Record<string, number>;
      by_type: Record<string, number>;
    }>>(`${BASE_PATH}/stats`);
    return data;
  },

  /**
   * Get benchmark results
   */
  async getBenchmarkResults(): Promise<APIResponse<unknown[]>> {
    const { data } = await apiClient.get<APIResponse<unknown[]>>(`${BASE_PATH}/benchmark/results`);
    return data;
  },
};

export default runsApi;
