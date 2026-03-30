/**
 * Authentication API methods
 */

import apiClient from './client';
import type { User, APIResponse } from '@/types';

const BASE_PATH = '/api/v1/auth';

/**
 * Login request payload
 */
export interface LoginRequest {
  email: string;
  password: string;
}

/**
 * Login response payload
 */
export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

/**
 * Refresh token response
 */
export interface RefreshResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

/**
 * Logout response
 */
export interface LogoutResponse {
  message: string;
}

export const authApi = {
  /**
   * User login
   * POST /api/v1/auth/login
   */
  async login(credentials: LoginRequest): Promise<LoginResponse> {
    const { data } = await apiClient.post<APIResponse<LoginResponse>>(
      `${BASE_PATH}/login`,
      credentials
    );

    if (!data.success || !data.data) {
      throw new Error(data.errors?.[0]?.message || 'Login failed');
    }

    return data.data;
  },

  /**
   * User logout
   * POST /api/v1/auth/logout
   */
  async logout(): Promise<LogoutResponse> {
    const { data } = await apiClient.post<APIResponse<LogoutResponse>>(
      `${BASE_PATH}/logout`
    );

    return data.data || { message: 'Logout successful' };
  },

  /**
   * Refresh access token
   * POST /api/v1/auth/refresh
   */
  async refreshToken(): Promise<RefreshResponse> {
    const { data } = await apiClient.post<APIResponse<RefreshResponse>>(
      `${BASE_PATH}/refresh`
    );

    if (!data.success || !data.data) {
      throw new Error(data.errors?.[0]?.message || 'Token refresh failed');
    }

    return data.data;
  },

  /**
   * Get current user info
   * GET /api/v1/auth/me
   */
  async getMe(): Promise<User> {
    const { data } = await apiClient.get<APIResponse<User>>(`${BASE_PATH}/me`);

    if (!data.success || !data.data) {
      throw new Error(data.errors?.[0]?.message || 'Failed to get user info');
    }

    return data.data;
  },
};

export default authApi;
