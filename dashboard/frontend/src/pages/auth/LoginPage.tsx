/**
 * Login Page
 */

import {
  Box,
  Card,
  Text,
  Title,
  TextInput,
  PasswordInput,
  Button,
  Group,
  Anchor,
  Divider,
  Paper,
  Center,
  Stack,
  Notification,
  Alert,
} from '@mantine/core';
import { useForm } from 'react-hook-form';
import { Link, useNavigate } from 'react-router-dom';
import { IconFlask, IconMail, IconLock, IconAlertCircle } from '@tabler/icons-react';
import { useAuthStore } from '@/stores/authStore';
import { useNotificationStore } from '@/stores/notificationStore';

interface LoginForm {
  email: string;
  password: string;
}

export default function LoginPage() {
  const { login, isLoading, error: authError } = useAuthStore();
  const addNotification = useNotificationStore((state) => state.addNotification);
  const navigate = useNavigate();

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginForm>({
    defaultValues: {
      email: '',
      password: '',
    },
  });

  const onSubmit = async (data: LoginForm) => {
    try {
      await login(data.email, data.password);
      addNotification({
        type: 'success',
        title: 'Login successful',
        message: 'Welcome to Lemonade Eval Dashboard',
      });
      navigate('/dashboard');
    } catch (error) {
      // Error is already set in auth store, but we can add a notification
      addNotification({
        type: 'error',
        title: 'Login failed',
        message: error instanceof Error ? error.message : 'Invalid credentials',
      });
    }
  };

  return (
    <Center w="100vw" h="100vh" bg="var(--mantine-color-gray-0)">
      <Paper w={420} p="xl" radius="md" shadow="lg">
        <Stack align="center" mb="xl">
          <IconFlask size={48} color="var(--mantine-color-blue-6)" />
          <Title order={2}>Lemonade Eval</Title>
          <Text c="dimmed" size="sm">
            Sign in to your dashboard
          </Text>
        </Stack>

        <Card padding="lg" radius="md" withBorder>
          <form onSubmit={handleSubmit(onSubmit)}>
            <Stack gap="md">
              {authError && (
                <Alert
                  variant="light"
                  color="red"
                  title="Authentication Error"
                  icon={<IconAlertCircle />}
                >
                  {authError}
                </Alert>
              )}

              <TextInput
                {...register('email', {
                  required: 'Email is required',
                  pattern: {
                    value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                    message: 'Invalid email address',
                  },
                })}
                label="Email"
                placeholder="you@example.com"
                leftSection={<IconMail size={16} />}
                error={errors.email?.message}
                size="md"
                disabled={isLoading}
              />

              <PasswordInput
                {...register('password', {
                  required: 'Password is required',
                  minLength: {
                    value: 6,
                    message: 'Password must be at least 6 characters',
                  },
                })}
                label="Password"
                placeholder="Enter your password"
                leftSection={<IconLock size={16} />}
                error={errors.password?.message}
                size="md"
                disabled={isLoading}
              />

              <Button
                type="submit"
                fullWidth
                size="md"
                loading={isLoading}
              >
                Sign in
              </Button>
            </Stack>
          </form>

          <Divider label="Demo Account" labelPosition="center" my="lg" />

          <Stack gap="xs">
            <Text size="xs" c="dimmed" ta="center">
              For demo purposes, you can use any email and password.
            </Text>
          </Stack>
        </Card>

        <Text size="xs" c="dimmed" ta="center" mt="lg">
          By signing in, you agree to our{' '}
          <Anchor component={Link} to="#" size="xs">
            Terms of Service
          </Anchor>{' '}
          and{' '}
          <Anchor component={Link} to="#" size="xs">
            Privacy Policy
          </Anchor>
        </Text>
      </Paper>
    </Center>
  );
}
