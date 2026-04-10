/**
 * Tests for Authentication Store
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { act } from '@testing-library/react';
import { useAuthStore } from '../stores/authStore';
import { authApi } from '../api/auth';
import { setAuthToken, clearAuthToken, getAuthToken } from '../api/client';

// Mock auth API
vi.mock('../api/auth', () => ({
  authApi: {
    login: vi.fn(),
    logout: vi.fn(),
    refreshToken: vi.fn(),
    getMe: vi.fn(),
  },
}));

// Mock client utilities
vi.mock('../api/client', () => ({
  setAuthToken: vi.fn(),
  clearAuthToken: vi.fn(),
  getAuthToken: vi.fn(),
  setApiKey: vi.fn(),
  clearApiKey: vi.fn(),
  getApiKey: vi.fn(),
  default: {},
}));

describe('Auth Store', () => {
  const mockUser = {
    id: 'test-user-123',
    email: 'test@example.com',
    name: 'Test User',
    role: 'editor' as const,
    is_active: true,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  };

  const mockToken = 'test-jwt-token-xyz';

  beforeEach(() => {
    vi.clearAllMocks();
    // Reset store to initial state
    useAuthStore.setState({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Initial State', () => {
    it('should have correct initial state', () => {
      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.token).toBeNull();
      expect(state.isAuthenticated).toBe(false);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBeNull();
    });
  });

  describe('Login', () => {
    it('should set loading state during login', async () => {
      vi.mocked(authApi.login).mockImplementation(
        () => new Promise((resolve) => setTimeout(resolve, 100))
      );

      const { login, isLoading } = useAuthStore.getState();

      // Start login (don't await yet)
      const loginPromise = login('test@example.com', 'password123');

      // Check loading state
      expect(useAuthStore.getState().isLoading).toBe(true);

      await loginPromise;
    });

    it('should successfully login and update state', async () => {
      const mockLoginResponse = {
        access_token: mockToken,
        token_type: 'bearer',
        expires_in: 1800,
        user: mockUser,
      };

      vi.mocked(authApi.login).mockResolvedValue(mockLoginResponse);

      const { login } = useAuthStore.getState();

      await act(async () => {
        await login('test@example.com', 'password123');
      });

      const state = useAuthStore.getState();
      expect(state.user).toEqual(mockUser);
      expect(state.token).toBe(mockToken);
      expect(state.isAuthenticated).toBe(true);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBeNull();
      expect(setAuthToken).toHaveBeenCalledWith(mockToken);
    });

    it('should handle login failure', async () => {
      const errorMessage = 'Invalid credentials';
      vi.mocked(authApi.login).mockRejectedValue(new Error(errorMessage));

      const { login } = useAuthStore.getState();

      await expect(login('test@example.com', 'wrongpassword')).rejects.toThrow(
        errorMessage
      );

      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.token).toBeNull();
      expect(state.isAuthenticated).toBe(false);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBe(errorMessage);
    });

    it('should clear previous error on new login attempt', async () => {
      // Set initial error
      useAuthStore.setState({ error: 'Previous error' });

      const mockLoginResponse = {
        access_token: mockToken,
        token_type: 'bearer',
        expires_in: 1800,
        user: mockUser,
      };

      vi.mocked(authApi.login).mockResolvedValue(mockLoginResponse);

      const { login } = useAuthStore.getState();

      await act(async () => {
        await login('test@example.com', 'password123');
      });

      expect(useAuthStore.getState().error).toBeNull();
    });
  });

  describe('Logout', () => {
    it('should clear auth state on logout', async () => {
      // Set authenticated state
      useAuthStore.setState({
        user: mockUser,
        token: mockToken,
        isAuthenticated: true,
      });

      vi.mocked(authApi.logout).mockResolvedValue({ message: 'Logout successful' });

      const { logout } = useAuthStore.getState();

      await act(async () => {
        await logout();
      });

      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.token).toBeNull();
      expect(state.isAuthenticated).toBe(false);
      expect(clearAuthToken).toHaveBeenCalled();
    });

    it('should clear state even if logout API fails', async () => {
      // Set authenticated state
      useAuthStore.setState({
        user: mockUser,
        token: mockToken,
        isAuthenticated: true,
      });

      vi.mocked(authApi.logout).mockRejectedValue(new Error('Network error'));

      const { logout } = useAuthStore.getState();

      await act(async () => {
        await logout();
      });

      // State should still be cleared
      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.token).toBeNull();
      expect(state.isAuthenticated).toBe(false);
      expect(clearAuthToken).toHaveBeenCalled();
    });
  });

  describe('Token Refresh', () => {
    it('should refresh token successfully', async () => {
      const newToken = 'new-jwt-token-abc';

      vi.mocked(authApi.refreshToken).mockResolvedValue({
        access_token: newToken,
        token_type: 'bearer',
        expires_in: 1800,
      });

      // Set initial state with old token
      useAuthStore.setState({
        user: mockUser,
        token: mockToken,
        isAuthenticated: true,
      });

      const { refreshToken } = useAuthStore.getState();

      await act(async () => {
        await refreshToken();
      });

      const state = useAuthStore.getState();
      expect(state.token).toBe(newToken);
      expect(setAuthToken).toHaveBeenCalledWith(newToken);
    });

    it('should clear auth state if refresh fails', async () => {
      vi.mocked(authApi.refreshToken).mockRejectedValue(new Error('Token expired'));

      // Set initial state
      useAuthStore.setState({
        user: mockUser,
        token: mockToken,
        isAuthenticated: true,
      });

      const { refreshToken } = useAuthStore.getState();

      await expect(refreshToken()).rejects.toThrow('Token expired');

      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.isAuthenticated).toBe(false);
      expect(clearAuthToken).toHaveBeenCalled();
    });
  });

  describe('Set User', () => {
    it('should update user and set isAuthenticated to true', () => {
      const { setUser } = useAuthStore.getState();

      act(() => {
        setUser(mockUser);
      });

      const state = useAuthStore.getState();
      expect(state.user).toEqual(mockUser);
      expect(state.isAuthenticated).toBe(true);
    });

    it('should set isAuthenticated to false when user is null', () => {
      useAuthStore.setState({ isAuthenticated: true });

      const { setUser } = useAuthStore.getState();

      act(() => {
        setUser(null);
      });

      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.isAuthenticated).toBe(false);
    });
  });

  describe('Set Token', () => {
    it('should update token in state', () => {
      const { setToken } = useAuthStore.getState();

      act(() => {
        setToken('new-token');
      });

      expect(useAuthStore.getState().token).toBe('new-token');
    });

    it('should clear token when set to null', () => {
      useAuthStore.setState({ token: 'existing-token' });

      const { setToken } = useAuthStore.getState();

      act(() => {
        setToken(null);
      });

      expect(useAuthStore.getState().token).toBeNull();
    });
  });

  describe('Clear Error', () => {
    it('should clear error state', () => {
      useAuthStore.setState({ error: 'Some error' });

      const { clearError } = useAuthStore.getState();

      act(() => {
        clearError();
      });

      expect(useAuthStore.getState().error).toBeNull();
    });
  });

  describe('Check Auth', () => {
    it('should return true when token and user exist', () => {
      useAuthStore.setState({
        user: mockUser,
        token: mockToken,
      });

      vi.mocked(getAuthToken).mockReturnValue(mockToken);

      const { checkAuth } = useAuthStore.getState();
      const result = checkAuth();

      expect(result).toBe(true);
      expect(useAuthStore.getState().isAuthenticated).toBe(true);
    });

    it('should return false when token does not exist', () => {
      useAuthStore.setState({
        user: mockUser,
        token: null,
      });

      vi.mocked(getAuthToken).mockReturnValue(null);

      const { checkAuth } = useAuthStore.getState();
      const result = checkAuth();

      expect(result).toBe(false);
      expect(useAuthStore.getState().isAuthenticated).toBe(false);
    });

    it('should return false when user does not exist', () => {
      useAuthStore.setState({
        user: null,
        token: mockToken,
      });

      const { checkAuth } = useAuthStore.getState();
      const result = checkAuth();

      expect(result).toBe(false);
    });
  });
});
