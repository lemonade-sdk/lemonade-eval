/**
 * Settings Page - API Keys and Preferences
 */

import {
  Group,
  Text,
  Title,
  Card,
  Grid,
  Button,
  Box,
  Switch,
  Select,
  Divider,
  Stack,
  Alert,
  Code,
  ActionIcon,
  Modal,
  TextInput,
  CopyButton,
  Tooltip,
  Badge,
} from '@mantine/core';
import { useState } from 'react';
import { useForm } from 'react-hook-form';
import {
  IconKey,
  IconPlus,
  IconTrash,
  IconCopy,
  IconCheck,
  IconMoon,
  IconSun,
  IconBell,
  IconClock,
  IconTable,
} from '@tabler/icons-react';
import { useUIStore } from '@/stores/uiStore';
import { useAuthStore } from '@/stores/authStore';
import { useNotificationStore } from '@/stores/notificationStore';

interface SettingsForm {
  notifications_enabled: boolean;
  refresh_interval: number;
  items_per_page: number;
}

export default function SettingsPage() {
  const { colorScheme, toggleColorScheme, refreshInterval, itemsPerPage, setRefreshInterval, setItemsPerPage } = useUIStore();
  const { user } = useAuthStore();
  const addNotification = useNotificationStore((state) => state.addNotification);

  const [apiKeys, setApiKeys] = useState<{ id: string; prefix: string; created: string }[]>([
    { id: '1', prefix: 'ledash_sk_demo_', created: new Date().toISOString() },
  ]);
  const [createKeyModalOpen, setCreateKeyModalOpen] = useState(false);
  const [newKeyName, setNewKeyName] = useState('');

  const handleCreateApiKey = () => {
    const newKey = {
      id: Date.now().toString(),
      prefix: `ledash_sk_${Math.random().toString(36).substring(2, 8)}_`,
      created: new Date().toISOString(),
    };
    setApiKeys([...apiKeys, newKey]);
    setCreateKeyModalOpen(false);
    setNewKeyName('');
    addNotification({
      type: 'success',
      title: 'API Key Created',
      message: 'Make sure to copy your API key. You won\'t be able to see it again!',
      duration: 10000,
    });
  };

  const handleDeleteApiKey = (id: string) => {
    setApiKeys(apiKeys.filter((key) => key.id !== id));
    addNotification({
      type: 'success',
      title: 'API Key Deleted',
    });
  };

  return (
    <Box>
      <Title order={2} mb="xl">
        Settings
      </Title>

      <Grid>
        <Grid.Col span={{ base: 12, lg: 6 }}>
          {/* Appearance Settings */}
          <Card padding="lg" radius="md" withBorder mb="md">
            <Group mb="md">
              <IconSun size={20} />
              <Title order={4}>Appearance</Title>
            </Group>
            <Stack gap="md">
              <Group justify="space-between">
                <div>
                  <Text fw={500}>Theme</Text>
                  <Text size="xs" c="dimmed">
                    Toggle between light and dark mode
                  </Text>
                </div>
                <ActionIcon
                  variant="outline"
                  size="lg"
                  onClick={toggleColorScheme}
                  aria-label={`Switch to ${colorScheme === 'light' ? 'dark' : 'light'} mode`}
                >
                  {colorScheme === 'light' ? <IconMoon size={18} /> : <IconSun size={18} />}
                </ActionIcon>
              </Group>
              <Divider />
              <Text size="sm" c="dimmed">
                Current theme: <Badge>{colorScheme}</Badge>
              </Text>
            </Stack>
          </Card>

          {/* Display Settings */}
          <Card padding="lg" radius="md" withBorder mb="md">
            <Group mb="md">
              <IconTable size={20} />
              <Title order={4}>Display</Title>
            </Group>
            <Stack gap="md">
              <Box>
                <Text size="sm" fw={600} mb="xs">
                  Items per page
                </Text>
                <Select
                  value={itemsPerPage.toString()}
                  onChange={(value) => setItemsPerPage(Number(value))}
                  data={[
                    { value: '10', label: '10' },
                    { value: '20', label: '20' },
                    { value: '50', label: '50' },
                    { value: '100', label: '100' },
                  ]}
                  w={150}
                />
                <Text size="xs" c="dimmed" mt="xs">
                  Number of items to display in tables
                </Text>
              </Box>
            </Stack>
          </Card>
        </Grid.Col>

        <Grid.Col span={{ base: 12, lg: 6 }}>
          {/* Notification Settings */}
          <Card padding="lg" radius="md" withBorder mb="md">
            <Group mb="md">
              <IconBell size={20} />
              <Title order={4}>Notifications</Title>
            </Group>
            <Stack gap="md">
              <Switch
                label="Enable notifications"
                description="Show toast notifications for events"
                defaultChecked
                size="md"
              />
              <Box>
                <Text size="sm" fw={600} mb="xs">
                  Refresh interval
                </Text>
                <Select
                  value={refreshInterval.toString()}
                  onChange={(value) => setRefreshInterval(Number(value))}
                  data={[
                    { value: '10000', label: '10 seconds' },
                    { value: '30000', label: '30 seconds' },
                    { value: '60000', label: '1 minute' },
                    { value: '300000', label: '5 minutes' },
                  ]}
                  w={150}
                />
                <Text size="xs" c="dimmed" mt="xs">
                  How often to refresh data from the server
                </Text>
              </Box>
            </Stack>
          </Card>

          {/* API Keys */}
          <Card padding="lg" radius="md" withBorder>
            <Group justify="space-between" mb="md">
              <Group>
                <IconKey size={20} />
                <Title order={4}>API Keys</Title>
              </Group>
              <Button
                size="sm"
                leftSection={<IconPlus size={16} />}
                onClick={() => setCreateKeyModalOpen(true)}
              >
                Create Key
              </Button>
            </Group>

            {apiKeys.length === 0 ? (
              <Text c="dimmed" ta="center" py="xl">
                No API keys yet. Create one to access the API programmatically.
              </Text>
            ) : (
              <Stack gap="md">
                {apiKeys.map((key) => (
                  <Card key={key.id} padding="sm" radius="md" withBorder>
                    <Group justify="space-between">
                      <Group>
                        <Code size="sm">{key.prefix}...</Code>
                        <CopyButton value={key.prefix}>
                          {({ copied, copy }) => (
                            <Tooltip label={copied ? 'Copied' : 'Copy'}>
                              <ActionIcon size="sm" variant="subtle" onClick={copy} aria-label="Copy API key">
                                <IconCopy size={14} />
                              </ActionIcon>
                            </Tooltip>
                          )}
                        </CopyButton>
                      </Group>
                      <Group gap="xs">
                        <Text size="xs" c="dimmed">
                          Created {new Date(key.created).toLocaleDateString()}
                        </Text>
                        <ActionIcon
                          size="sm"
                          color="red"
                          variant="subtle"
                          onClick={() => handleDeleteApiKey(key.id)}
                          aria-label="Delete API key"
                        >
                          <IconTrash size={14} />
                        </ActionIcon>
                      </Group>
                    </Group>
                  </Card>
                ))}
              </Stack>
            )}

            <Alert title="API Key Security" icon={<IconKey size={16} />} variant="light" mt="md">
              <Text size="sm">
                API keys provide full access to your dashboard. Never share your API key or commit
                it to version control. If a key is compromised, delete it immediately and create a
                new one.
              </Text>
            </Alert>
          </Card>
        </Grid.Col>
      </Grid>

      {/* Create API Key Modal */}
      <Modal
        opened={createKeyModalOpen}
        onClose={() => setCreateKeyModalOpen(false)}
        title="Create API Key"
      >
        <Stack gap="md">
          <TextInput
            label="Key Name"
            placeholder="My API Key"
            value={newKeyName}
            onChange={(e) => setNewKeyName(e.target.value)}
          />
          <Alert title="Important" icon={<IconCheck size={16} />} color="blue" variant="light">
            <Text size="sm">
              Your API key will only be shown once after creation. Make sure to copy it to a safe
              location.
            </Text>
          </Alert>
          <Group justify="flex-end">
            <Button variant="outline" onClick={() => setCreateKeyModalOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateApiKey}>Create API Key</Button>
          </Group>
        </Stack>
      </Modal>
    </Box>
  );
}
