/**
 * API Client Configuration
 * Base axios instance with interceptors for auth and error handling
 */

import axios, { AxiosError, AxiosInstance, InternalAxiosRequestConfig } from 'axios';
import { ApiErrorClass } from '@/types';

// API base URL from environment or default
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001';

// Request timeout in milliseconds
const REQUEST_TIMEOUT = 30000;

// In-memory token storage (more secure than localStorage)
// Tokens stored in memory are cleared when the page is refreshed
let authToken: string | null = null;
let apiKey: string | null = null;

/**
 * Set auth token in memory
 * For production, consider using httpOnly cookies set by the backend
 */
export const setAuthToken = (token: string): void => {
  authToken = token;
  // Fallback to sessionStorage if persistence across tabs is needed
  // sessionStorage is cleared when the browser/tab is closed
  try {
    sessionStorage.setItem('auth_token', token);
  } catch (e) {
    console.warn('sessionStorage not available');
  }
};

/**
 * Get auth token from memory or sessionStorage
 */
export const getAuthToken = (): string | null => {
  if (authToken) {
    return authToken;
  }
  try {
    authToken = sessionStorage.getItem('auth_token');
  } catch (e) {
    console.warn('sessionStorage not available');
  }
  return authToken;
};

/**
 * Clear auth token from memory and storage
 */
export const clearAuthToken = (): void => {
  authToken = null;
  try {
    sessionStorage.removeItem('auth_token');
  } catch (e) {
    console.warn('sessionStorage not available');
  }
};

/**
 * Set API key in memory
 */
export const setApiKey = (key: string): void => {
  apiKey = key;
  try {
    sessionStorage.setItem('api_key', key);
  } catch (e) {
    console.warn('sessionStorage not available');
  }
};

/**
 * Get API key from memory or sessionStorage
 */
export const getApiKey = (): string | null => {
  if (apiKey) {
    return apiKey;
  }
  try {
    apiKey = sessionStorage.getItem('api_key');
  } catch (e) {
    console.warn('sessionStorage not available');
  }
  return apiKey;
};

/**
 * Clear API key from memory and storage
 */
export const clearApiKey = (): void => {
  apiKey = null;
  try {
    sessionStorage.removeItem('api_key');
  } catch (e) {
    console.warn('sessionStorage not available');
  }
};

/**
 * Create axios instance with default configuration
 */
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: REQUEST_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, // Enable cookie sending for secure auth
});

/**
 * Request interceptor - adds auth token to requests
 */
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Skip adding auth token for auth endpoints (login, register, etc.)
    const authEndpoints = ['/api/v1/auth/login', '/api/v1/auth/register', '/api/v1/auth/refresh'];
    const isAuthEndpoint = authEndpoints.some(endpoint =>
      config.url?.includes(endpoint)
    );

    if (!isAuthEndpoint) {
      // Get token from memory/sessionStorage (more secure than localStorage)
      const token = getAuthToken();

      if (token && config.headers) {
        config.headers.Authorization = `Bearer ${token}`;
      }

      // Add API key if available (alternative auth method)
      const key = getApiKey();
      if (key && config.headers) {
        config.headers['X-API-Key'] = key;
      }
    }

    return config;
  },
  (error: AxiosError) => {
    return Promise.reject(error);
  }
);

/**
 * Response interceptor - handles errors globally
 */
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    // Transform axios errors to our custom ApiErrorClass
    const status = error.response?.status || 500;
    const data = error.response?.data as {
      detail?: string;
      message?: string;
      code?: string;
      error_code?: string;
      errors?: Array<{ code: string; message: string }>;
    } | undefined;

    let message = 'An unexpected error occurred';
    let errorCode: string | undefined;

    if (data) {
      // Extract error code from response
      errorCode = data.code || data.error_code;

      // Get detail or message — handle FastAPI array detail format
      if (Array.isArray(data.detail)) {
        message = data.detail.map((d: { msg?: string }) => d.msg || String(d)).join('; ');
      } else if (typeof data.detail === 'string') {
        message = data.detail;
      } else if (data.message) {
        message = data.message;
      } else if (error.message) {
        message = error.message;
      }

      // Handle FastAPI validation errors
      if (Array.isArray(data.errors) && data.errors.length > 0) {
        errorCode = data.errors[0].code || errorCode;
      }
    }

    // Generate error code if not provided
    if (!errorCode) {
      errorCode = `ERR${status}`;
    }

    // Handle 401 Unauthorized - clear auth token
    if (status === 401) {
      clearAuthToken();
      // Dispatch custom event for auth state management
      window.dispatchEvent(new CustomEvent('auth-required'));
    }

    const apiError = new ApiErrorClass(message, status, errorCode, data);
    return Promise.reject(apiError);
  }
);

export default apiClient;
