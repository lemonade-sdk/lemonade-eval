/**
 * Test utilities for React components
 */

import { render, RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MantineProvider, createTheme } from '@mantine/core';
import { ReactNode } from 'react';

// Create a test query client
const createTestQueryClient = () => {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: 0,
      },
    },
  });
};

// Create test theme
const testTheme = createTheme({
  primaryColor: 'blue',
});

interface TestWrapperProps {
  children: ReactNode;
}

function TestWrapper({ children }: TestWrapperProps) {
  return (
    <QueryClientProvider client={createTestQueryClient()}>
      <MantineProvider theme={testTheme}>
        {children}
      </MantineProvider>
    </QueryClientProvider>
  );
}

/**
 * Custom render function with providers
 */
function customRender(ui: ReactNode, options?: Omit<RenderOptions, 'wrapper'>) {
  return render(ui, { wrapper: TestWrapper, ...options });
}

// Re-export everything from testing-library
export * from '@testing-library/react';

// Override render with custom one
export { customRender as render };
