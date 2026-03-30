/**
 * API methods for Import operations
 */

import apiClient from './client';
import type { ImportRequest, ImportJobStatus, ImportScanResult, APIResponse } from '@/types';

const BASE_PATH = '/api/v1/import';

export const importApi = {
  /**
   * Import YAML files from a cache directory
   */
  async importYaml(request: ImportRequest): Promise<APIResponse<{ job_id: string }>> {
    const { data } = await apiClient.post<APIResponse<{ job_id: string }>>(`${BASE_PATH}/yaml`, request);
    return data;
  },

  /**
   * Get status of an import job
   */
  async getImportStatus(jobId: string): Promise<APIResponse<ImportJobStatus>> {
    const { data } = await apiClient.get<APIResponse<ImportJobStatus>>(`${BASE_PATH}/status/${jobId}`);
    return data;
  },

  /**
   * Scan a cache directory
   */
  async scanCacheDirectory(cacheDir: string): Promise<APIResponse<ImportScanResult>> {
    const { data } = await apiClient.post<APIResponse<ImportScanResult>>(`${BASE_PATH}/scan`, null, {
      params: { cache_dir: cacheDir },
    });
    return data;
  },

  /**
   * List all import jobs
   */
  async listImportJobs(): Promise<APIResponse<ImportJobStatus[]>> {
    const { data } = await apiClient.get<APIResponse<ImportJobStatus[]>>(`${BASE_PATH}/jobs`);
    return data;
  },
};

export default importApi;
