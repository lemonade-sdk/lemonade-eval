/**
 * API methods for Metrics
 */

import apiClient from './client';
import type { Metric, MetricCreate, MetricListResponse, MetricTrend, APIResponse, MetricComparison } from '@/types';

const BASE_PATH = '/api/v1/metrics';

export interface ListMetricsParams {
  run_id?: string | null;
  category?: string | null;
  name?: string | null;
  page?: number;
  per_page?: number;
}

export const metricsApi = {
  /**
   * List metrics with pagination and filtering
   */
  async listMetrics(params?: ListMetricsParams): Promise<MetricListResponse> {
    const { data } = await apiClient.get<MetricListResponse>(BASE_PATH, { params });
    return data;
  },

  /**
   * Get a specific metric by ID
   */
  async getMetric(metricId: string): Promise<APIResponse<Metric>> {
    const { data } = await apiClient.get<APIResponse<Metric>>(`${BASE_PATH}/${metricId}`);
    return data;
  },

  /**
   * Create a new metric
   */
  async createMetric(metricData: MetricCreate): Promise<APIResponse<Metric>> {
    const { data } = await apiClient.post<APIResponse<Metric>>(BASE_PATH, metricData);
    return data;
  },

  /**
   * Create multiple metrics in bulk
   */
  async createMetricsBulk(metrics: MetricCreate[]): Promise<APIResponse<Metric[]>> {
    const { data } = await apiClient.post<APIResponse<Metric[]>>(`${BASE_PATH}/bulk`, { metrics });
    return data;
  },

  /**
   * Delete a metric
   */
  async deleteMetric(metricId: string): Promise<APIResponse<{ message: string }>> {
    const { data } = await apiClient.delete<APIResponse<{ message: string }>>(`${BASE_PATH}/${metricId}`);
    return data;
  },

  /**
   * Get aggregated metrics across runs
   */
  async getAggregateMetrics(params?: {
    model_id?: string | null;
    run_type?: string | null;
    category?: string | null;
    metric_name?: string | null;
  }): Promise<APIResponse<unknown[]>> {
    const { data } = await apiClient.get<APIResponse<unknown[]>>(`${BASE_PATH}/aggregate`, { params });
    return data;
  },

  /**
   * Get metric trends over time
   */
  async getMetricTrends(params: {
    model_id: string;
    metric_name: string;
    limit?: number;
  }): Promise<APIResponse<MetricTrend[]>> {
    const { data } = await apiClient.get<APIResponse<MetricTrend[]>>(`${BASE_PATH}/trends`, { params });
    return data;
  },

  /**
   * Compare metrics across multiple runs
   */
  async compareMetrics(runIds: string[], categories?: string[]): Promise<APIResponse<MetricComparison>> {
    const { data } = await apiClient.get<APIResponse<MetricComparison>>(`${BASE_PATH}/compare`, {
      params: {
        run_ids: runIds.join(','),
        categories: categories?.join(','),
      },
    });
    return data;
  },

  /**
   * Get performance metrics for a run
   */
  async getPerformanceMetrics(runId: string): Promise<APIResponse<Record<string, unknown>>> {
    const { data } = await apiClient.get<APIResponse<Record<string, unknown>>>(
      `${BASE_PATH}/performance/${runId}`
    );
    return data;
  },
};

export default metricsApi;
