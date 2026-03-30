/**
 * Import Page - YAML Import UI with Progress
 */

import {
  Group,
  Text,
  Title,
  Card,
  Grid,
  Button,
  Box,
  TextInput,
  Switch,
  Progress,
  Badge,
  Timeline,
  Alert,
  Stack,
  Divider,
  Code,
  Skeleton,
} from '@mantine/core';
import { useState } from 'react';
import { useForm } from 'react-hook-form';
import {
  IconUpload,
  IconFolder,
  IconCheck,
  IconX,
  IconClock,
  IconAlertCircle,
} from '@tabler/icons-react';
import { useImportYaml, useImportStatus, useScanCacheDirectory } from '@/hooks/useImport';
import { LoadingSpinner, ErrorDisplay } from '@/components/common';
import { useWebSocket } from '@/hooks/useWebSocket';

interface ImportForm {
  cache_dir: string;
  skip_duplicates: boolean;
  dry_run: boolean;
}

export default function ImportPage() {
  const [lastJobId, setLastJobId] = useState<string | null>(null);
  const [scanResults, setScanResults] = useState<string[] | null>(null);

  const { register, handleSubmit, watch, setValue } = useForm<ImportForm>({
    defaultValues: {
      cache_dir: '',
      skip_duplicates: true,
      dry_run: false,
    },
  });

  const importMutation = useImportYaml();
  const scanMutation = useScanCacheDirectory();

  const cacheDir = watch('cache_dir');
  const dryRun = watch('dry_run');

  const { data: jobStatusData, isLoading: statusLoading } = useImportStatus(
    lastJobId || '',
    !!lastJobId
  );

  const jobStatus = jobStatusData?.data;
  const isCompleted = jobStatus?.status === 'completed';
  const isFailed = jobStatus?.status === 'failed';
  const isRunning = jobStatus?.status === 'running';

  const handleScan = async () => {
    if (!cacheDir) return;

    try {
      const result = await scanMutation.mutateAsync(cacheDir);
      setScanResults(result.data?.files || []);
    } catch (error) {
      setScanResults(null);
    }
  };

  const onSubmit = async (data: ImportForm) => {
    try {
      const result = await importMutation.mutateAsync(data);
      setLastJobId(result.data?.job_id || null);
    } catch (error) {
      console.error('Import failed:', error);
    }
  };

  const progress = jobStatus
    ? jobStatus.total_files > 0
      ? (jobStatus.processed_files / jobStatus.total_files) * 100
      : 0
    : 0;

  return (
    <Box>
      <Group justify="space-between" mb="xl">
        <Title order={2}>Import Evaluation Results</Title>
      </Group>

      <Grid>
        <Grid.Col span={{ base: 12, lg: 6 }}>
          {/* Import Form */}
          <Card padding="lg" radius="md" withBorder>
            <Title order={4} mb="md">
              Import Configuration
            </Title>
            <form onSubmit={handleSubmit(onSubmit)}>
              <Stack gap="md">
                <Box>
                  <Text size="sm" fw={600} mb="xs">
                    Cache Directory
                  </Text>
                  <Group gap="xs">
                    <TextInput
                      {...register('cache_dir')}
                      placeholder="~/.cache/lemonade"
                      style={{ flex: 1 }}
                      leftSection={<IconFolder size={16} />}
                      disabled={isRunning}
                    />
                    <Button
                      type="button"
                      variant="outline"
                      onClick={handleScan}
                      disabled={!cacheDir || scanMutation.isPending || isRunning}
                      leftSection={scanMutation.isPending ? <IconClock size={16} /> : <IconFolder size={16} />}
                    >
                      {scanMutation.isPending ? 'Scanning...' : 'Scan'}
                    </Button>
                  </Group>
                  <Text size="xs" c="dimmed" mt="xs">
                    Path to your lemonade-eval cache directory containing YAML evaluation results
                  </Text>
                </Box>

                {scanResults && (
                  <Alert
                    title={`Found ${scanResults.length} files`}
                    icon={<IconCheck size={16} />}
                    color="blue"
                    variant="light"
                  >
                    {scanResults.slice(0, 5).map((f) => (
                      <Code key={f} display="block" mb="xs">
                        {f.split('/').pop()}
                      </Code>
                    ))}
                    {scanResults.length > 5 && (
                      <Text size="xs" c="dimmed">
                        ...and {scanResults.length - 5} more files
                      </Text>
                    )}
                  </Alert>
                )}

                <Switch
                  {...register('skip_duplicates')}
                  label="Skip duplicates"
                  description="Skip runs that already exist in the database"
                  disabled={isRunning}
                />

                <Switch
                  {...register('dry_run')}
                  label="Dry run"
                  description="Only scan files without importing"
                  disabled={isRunning}
                />

                <Divider />

                <Group justify="flex-end">
                  <Button
                    type="submit"
                    disabled={!cacheDir || isRunning}
                    loading={importMutation.isPending}
                    leftSection={<IconUpload size={16} />}
                  >
                    {dryRun ? 'Scan Files' : 'Start Import'}
                  </Button>
                </Group>
              </Stack>
            </form>
          </Card>
        </Grid.Col>

        <Grid.Col span={{ base: 12, lg: 6 }}>
          {/* Import Status */}
          <Card padding="lg" radius="md" withBorder>
            <Title order={4} mb="md">
              Import Status
            </Title>

            {!lastJobId ? (
              <Text c="dimmed" ta="center" py="xl">
                Start an import to see progress here
              </Text>
            ) : statusLoading || !jobStatus ? (
              <Skeleton height={200} />
            ) : (
              <Stack gap="md">
                <Group justify="space-between">
                  <Text size="sm" fw={600}>
                    Job ID
                  </Text>
                  <Code>{jobStatus.job_id.slice(0, 8)}...</Code>
                </Group>

                <Group justify="space-between">
                  <Text size="sm" fw={600}>
                    Status
                  </Text>
                  <Badge
                    color={
                      isCompleted ? 'green' : isFailed ? 'red' : isRunning ? 'blue' : 'gray'
                    }
                    variant="light"
                  >
                    {jobStatus.status.toUpperCase()}
                  </Badge>
                </Group>

                {jobStatus.total_files > 0 && (
                  <>
                    <Progress
                      value={progress}
                      size="lg"
                      color={isFailed ? 'red' : isCompleted ? 'green' : 'blue'}
                      striped={isRunning}
                      animated={isRunning}
                    />
                    <Group justify="space-between">
                      <Text size="xs" c="dimmed">
                        {jobStatus.processed_files} / {jobStatus.total_files} files
                      </Text>
                      <Text size="xs" c="dimmed">
                        {progress.toFixed(0)}%
                      </Text>
                    </Group>
                  </>
                )}

                <Grid>
                  <Grid.Col span={4}>
                    <Box ta="center">
                      <Text size="xs" c="dimmed">
                        Imported
                      </Text>
                      <Text size="xl" fw={700} c="green.6">
                        {jobStatus.imported_runs}
                      </Text>
                    </Box>
                  </Grid.Col>
                  <Grid.Col span={4}>
                    <Box ta="center">
                      <Text size="xs" c="dimmed">
                        Skipped
                      </Text>
                      <Text size="xl" fw={700} c="blue.6">
                        {jobStatus.skipped_duplicates}
                      </Text>
                    </Box>
                  </Grid.Col>
                  <Grid.Col span={4}>
                    <Box ta="center">
                      <Text size="xs" c="dimmed">
                        Errors
                      </Text>
                      <Text size="xl" fw={700} c="red.6">
                        {jobStatus.errors?.length || 0}
                      </Text>
                    </Box>
                  </Grid.Col>
                </Grid>

                {jobStatus.errors && jobStatus.errors.length > 0 && (
                  <Alert
                    title="Import Errors"
                    icon={<IconAlertCircle size={16} />}
                    color="red"
                    variant="light"
                  >
                    {jobStatus.errors.map((error: string, idx: number) => (
                      <Code key={idx} display="block" mb="xs">
                        {error}
                      </Code>
                    ))}
                  </Alert>
                )}

                {isCompleted && (
                  <Alert
                    title="Import Completed"
                    icon={<IconCheck size={16} />}
                    color="green"
                    variant="light"
                  >
                    Successfully imported {jobStatus.imported_runs} evaluation runs
                    {jobStatus.skipped_duplicates > 0 && (
                      <>
                        {' '}
                        ({jobStatus.skipped_duplicates} duplicates skipped)
                      </>
                    )}
                  </Alert>
                )}

                {isFailed && (
                  <Alert
                    title="Import Failed"
                    icon={<IconX size={16} />}
                    color="red"
                    variant="light"
                  >
                    {jobStatus.error || 'An unknown error occurred'}
                  </Alert>
                )}
              </Stack>
            )}
          </Card>
        </Grid.Col>
      </Grid>
    </Box>
  );
}
