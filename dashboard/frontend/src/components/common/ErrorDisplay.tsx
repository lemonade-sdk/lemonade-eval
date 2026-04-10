/**
 * Error Display Component
 */

import { Center, Text, Button, Stack, Icon, Box } from '@mantine/core';
import { IconAlertCircle, IconRefresh } from '@tabler/icons-react';

interface ErrorDisplayProps {
  message?: string;
  onRetry?: () => void;
  fullScreen?: boolean;
}

export function ErrorDisplay({ message, onRetry, fullScreen = false }: ErrorDisplayProps) {
  const content = (
    <Stack align="center" gap="sm">
      <IconAlertCircle size={48} color="var(--mantine-color-red-6)" stroke={1.5} />
      <Text fw={600} size="lg">
        Something went wrong
      </Text>
      {message && (
        <Text size="sm" c="dimmed" ta="center" maw={400}>
          {message}
        </Text>
      )}
      {onRetry && (
        <Button leftSection={<IconRefresh size={16} />} onClick={onRetry} variant="outline">
          Try again
        </Button>
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
      <Box>{content}</Box>
    </Center>
  );
}

export default ErrorDisplay;
