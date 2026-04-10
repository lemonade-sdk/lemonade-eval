/**
 * Main application entry point
 */

import React, { useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import { createTheme, MantineProvider, useMantineColorScheme } from '@mantine/core';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { BrowserRouter } from 'react-router-dom';
import { Notifications } from '@mantine/notifications';
import '@mantine/core/styles.css';
import '@mantine/dates/styles.css';
import '@mantine/notifications/styles.css';

import App from './App';
import { useUIStore } from './stores/uiStore';

// Create theme
const theme = createTheme({
  primaryColor: 'blue',
  colors: {
    blue: [
      '#e6f0ff',
      '#c9deff',
      '#93c2ff',
      '#54a3ff',
      '#208aff',
      '#0078eb',
      '#0066cc',
      '#0052b3',
      '#003d99',
      '#002680',
    ],
  },
  fontFamily: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  headings: {
    fontFamily: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  defaultRadius: 'md',
  cursorType: 'pointer',
  components: {
    Button: {
      defaultProps: {
        fw: 500,
      },
    },
    Card: {
      defaultProps: {
        shadow: 'sm',
        withBorder: true,
      },
    },
  },
});

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 30000, // 30 seconds
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 1,
    },
  },
});

// Get initial color scheme
const getInitialColorScheme = () => {
  // Check if we have a persisted value
  const stored = typeof window !== 'undefined' ? localStorage.getItem('ui-storage') : null;
  if (stored) {
    try {
      const parsed = JSON.parse(stored);
      if (parsed.state?.colorScheme) {
        return parsed.state.colorScheme;
      }
    } catch {
      // Ignore parse errors
    }
  }
  // Fall back to system preference
  if (typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    return 'dark';
  }
  return 'light';
};

// Component to handle color scheme synchronization
function ColorSchemeSync() {
  const { setColorScheme } = useMantineColorScheme();
  const uiColorScheme = useUIStore((state) => state.colorScheme);

  // Sync Mantine color scheme with UI store on mount and when store changes
  useEffect(() => {
    setColorScheme(uiColorScheme);
  }, [uiColorScheme, setColorScheme]);

  return null;
}

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <MantineProvider theme={theme} defaultColorScheme="auto">
      <ColorSchemeSync />
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Notifications position="top-right" />
          <App />
        </BrowserRouter>
        <ReactQueryDevtools initialIsOpen={false} />
      </QueryClientProvider>
    </MantineProvider>
  </React.StrictMode>
);
