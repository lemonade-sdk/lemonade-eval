/**
 * Authentication Store using Zustand
 * Manages user authentication state and actions
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { User } from '@/types';
import { authApi } from '@/api/auth';
import { setAuthToken, clearAuthToken, getAuthToken } from '@/api/client';
import { parseApiError } from '@/utils/errors';

interface AuthState {
  // State
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  errorCode: string | null;

  // Actions
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  setUser: (user: User | null) => void;
  setToken: (token: string | null) => void;
  clearError: () => void;
  checkAuth: () => boolean;
  refreshToken: () => Promise<void>;
}

const initialState = {
  user: null,
  token: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,
  errorCode: null,
};

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      ...initialState,

      login: async (email: string, password: string) => {
        set({ isLoading: true, error: null, errorCode: null });
        try {
          // Call real API login endpoint
          const response = await authApi.login({ email, password });

          // Store token in sessionStorage via utility
          setAuthToken(response.access_token);

          // Update state with user info and token
          set({
            user: response.user,
            token: response.access_token,
            isAuthenticated: true,
            isLoading: false,
            error: null,
            errorCode: null,
          });
        } catch (error) {
          const errorInfo = parseApiError(error);
          set({
            error: errorInfo.userMessage,
            errorCode: errorInfo.code,
            isLoading: false,
          });
          throw error;
        }
      },

      logout: async () => {
        try {
          // Call logout API endpoint (optional, for token invalidation)
          await authApi.logout().catch(() => {
            // Ignore logout API errors - still clear local state
            console.warn('Logout API call failed, but clearing local state');
          });
        } finally {
          // Always clear local auth state
          clearAuthToken();
          set(initialState);
        }
      },

      refreshToken: async () => {
        try {
          const response = await authApi.refreshToken();
          setAuthToken(response.access_token);
          set({ token: response.access_token });
        } catch (error) {
          // Token refresh failed - clear auth state
          clearAuthToken();
          set(initialState);
          throw error;
        }
      },

      setUser: (user: User | null) => {
        set({ user, isAuthenticated: !!user });
      },

      setToken: (token: string | null) => {
        set({ token });
      },

      clearError: () => {
        set({ error: null, errorCode: null });
      },

      checkAuth: () => {
        // Check for token in sessionStorage
        const token = getAuthToken();
        const { user } = get();

        if (token && user) {
          set({ isAuthenticated: true, token });
          return true;
        }

        set({ isAuthenticated: false });
        return false;
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ user: state.user, token: state.token }),
    }
  )
);

export default useAuthStore;
