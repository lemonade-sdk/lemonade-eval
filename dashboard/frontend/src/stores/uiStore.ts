/**
 * UI State Store using Zustand
 * Manages UI state like theme, sidebar, modals
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UIState {
  // Theme
  colorScheme: 'light' | 'dark';
  toggleColorScheme: () => void;
  setColorScheme: (scheme: 'light' | 'dark') => void;

  // Sidebar
  sidebarOpened: boolean;
  toggleSidebar: () => void;
  setSidebarOpened: (opened: boolean) => void;

  // Notifications
  notificationsEnabled: boolean;
  setNotificationsEnabled: (enabled: boolean) => void;

  // Refresh interval for polling (ms)
  refreshInterval: number;
  setRefreshInterval: (interval: number) => void;

  // Items per page for tables
  itemsPerPage: number;
  setItemsPerPage: (count: number) => void;

  // Default cache directory for YAML imports
  cacheDir: string;
  setCacheDir: (dir: string) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      // Theme - default to light, will be synced with Mantine
      colorScheme: 'light',
      toggleColorScheme: () =>
        set((state) => ({ colorScheme: state.colorScheme === 'light' ? 'dark' : 'light' })),
      setColorScheme: (scheme) => set({ colorScheme: scheme }),

      // Sidebar
      sidebarOpened: true,
      toggleSidebar: () => set((state) => ({ sidebarOpened: !state.sidebarOpened })),
      setSidebarOpened: (opened) => set({ sidebarOpened: opened }),

      // Notifications
      notificationsEnabled: true,
      setNotificationsEnabled: (enabled) => set({ notificationsEnabled: enabled }),

      // Refresh interval (default 30 seconds)
      refreshInterval: 30000,
      setRefreshInterval: (interval) => set({ refreshInterval: interval }),

      // Items per page (default 20)
      itemsPerPage: 20,
      setItemsPerPage: (count) => set({ itemsPerPage: count }),

      // Cache directory (default matches lemonade-eval default)
      cacheDir: '~/.cache/lemonade',
      setCacheDir: (dir) => set({ cacheDir: dir }),
    }),
    {
      name: 'ui-storage',
      partialize: (state) => ({
        colorScheme: state.colorScheme,
        sidebarOpened: state.sidebarOpened,
        notificationsEnabled: state.notificationsEnabled,
        refreshInterval: state.refreshInterval,
        itemsPerPage: state.itemsPerPage,
        cacheDir: state.cacheDir,
      }),
    }
  )
);

export default useUIStore;
