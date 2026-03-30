/**
 * API methods for Models
 */

import apiClient from './client';
import type { Model, ModelCreate, ModelUpdate, ModelListResponse, APIResponse } from '@/types';

const BASE_PATH = '/api/v1/models';

export const modelsApi = {
  /**
   * List models with pagination and filtering
   */
  async listModels(params?: {
    page?: number;
    per_page?: number;
    search?: string | null;
    family?: string | null;
    model_type?: string | null;
  }): Promise<ModelListResponse> {
    const { data } = await apiClient.get<ModelListResponse>(BASE_PATH, { params });
    return data;
  },

  /**
   * Get a specific model by ID
   */
  async getModel(modelId: string): Promise<APIResponse<Model>> {
    const { data } = await apiClient.get<APIResponse<Model>>(`${BASE_PATH}/${modelId}`);
    return data;
  },

  /**
   * Create a new model
   */
  async createModel(modelData: ModelCreate): Promise<APIResponse<Model>> {
    const { data } = await apiClient.post<APIResponse<Model>>(BASE_PATH, modelData);
    return data;
  },

  /**
   * Update an existing model
   */
  async updateModel(modelId: string, modelData: ModelUpdate): Promise<APIResponse<Model>> {
    const { data } = await apiClient.put<APIResponse<Model>>(`${BASE_PATH}/${modelId}`, modelData);
    return data;
  },

  /**
   * Delete a model
   */
  async deleteModel(modelId: string): Promise<APIResponse<{ message: string }>> {
    const { data } = await apiClient.delete<APIResponse<{ message: string }>>(`${BASE_PATH}/${modelId}`);
    return data;
  },

  /**
   * Get model versions
   */
  async getModelVersions(modelId: string): Promise<APIResponse<unknown[]>> {
    const { data } = await apiClient.get<APIResponse<unknown[]>>(`${BASE_PATH}/${modelId}/versions`);
    return data;
  },

  /**
   * Get runs for a model
   */
  async getModelRuns(modelId: string, limit?: number): Promise<APIResponse<Run[]>> {
    const { data } = await apiClient.get<APIResponse<Run[]>>(`${BASE_PATH}/${modelId}/runs`, {
      params: { limit },
    });
    return data;
  },

  /**
   * List unique model families
   */
  async listFamilies(): Promise<APIResponse<string[]>> {
    const { data } = await apiClient.get<APIResponse<string[]>>(`${BASE_PATH}/families/list`);
    return data;
  },
};

// Import Run type to avoid circular dependency
import type { Run } from '@/types';

export default modelsApi;
