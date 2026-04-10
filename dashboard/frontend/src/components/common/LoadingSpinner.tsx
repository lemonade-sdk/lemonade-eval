/**
 * Loading Spinner Component
 */

import { Center, Loader, Text, Stack } from '@mantine/core';

interface LoadingSpinnerProps {
  message?: string;
  size?: 'sm' | 'md' | 'lg';
  fullScreen?: boolean;
}

export function LoadingSpinner({ message, size = 'md', fullScreen = false }: LoadingSpinnerProps) {
  const content = (
    <Stack align="center" gap="sm">
      <Loader size={size} />
      {message && (
        <Text size="sm" c="dimmed">
          {message}
        </Text>
      )}
    </Stack>
  );

  if (fullScreen) {
    return (
      <Center w="100vw" h="100vh">
        {content}
      </Center>
    );
  }

  return (
    <Center w="100%" h="100%" p="xl">
      {content}
    </Center>
  );
}

export default LoadingSpinner;
