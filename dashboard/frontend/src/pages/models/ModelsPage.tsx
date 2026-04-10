/**
 * Models Page - List and Search Models
 */

import {
  Group,
  Text,
  Title,
  Card,
  Grid,
  Badge,
  Button,
  TextInput,
  Select,
  Box,
  Skeleton,
  Modal,
  Stack,
  Divider,
} from '@mantine/core';
import { useForm } from 'react-hook-form';
import { useState } from 'react';
import { IconSearch, IconPlus, IconTrash } from '@tabler/icons-react';
import { useNavigate } from 'react-router-dom';
import { useModels, useDeleteModel, useModelFamilies, useCreateModel } from '@/hooks/useModels';
import { LoadingSpinner, ErrorDisplay, DataTable, StatusBadge } from '@/components/common';
import { formatDateTime, extractModelFamily, parseQuantization } from '@/utils';
import { ColumnDef } from '@tanstack/react-table';
import { ActionIcon } from '@mantine/core';
import type { Model, ModelCreate } from '@/types';

export default function ModelsPage() {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [selectedFamily, setSelectedFamily] = useState<string | null>(null);
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [addModalOpen, setAddModalOpen] = useState(false);

  const { data: modelsData, isLoading, error } = useModels({
    page: 1,
    per_page: 50,
    search: search || null,
    family: selectedFamily,
    model_type: selectedType,
  });

  const { data: familiesData } = useModelFamilies();
  const deleteModel = useDeleteModel();
  const createModel = useCreateModel();

  const { register, handleSubmit, reset, formState: { errors } } = useForm<ModelCreate>({
    defaultValues: { name: '', checkpoint: '', model_type: 'llm' },
  });

  const handleAddModel = (data: ModelCreate) => {
    createModel.mutate(data, {
      onSuccess: () => {
        setAddModalOpen(false);
        reset();
      },
    });
  };

  const models = modelsData?.data || [];
  const families = ['All', ...(familiesData?.data || [])].map((f) => ({ value: f, label: f }));

  const handleDelete = () => {
    if (selectedModel) {
      deleteModel.mutate(selectedModel.id);
      setDeleteModalOpen(false);
      setSelectedModel(null);
    }
  };

  const columns: ColumnDef<Model>[] = [
    {
      accessorKey: 'name',
      header: 'Name',
      cell: ({ row }) => (
        <div>
          <Text fw={600} size="sm">
            {row.original.name}
          </Text>
          <Text size="xs" c="dimmed">
            {row.original.checkpoint}
          </Text>
        </div>
      ),
    },
    {
      accessorKey: 'family',
      header: 'Family',
      cell: ({ row }) => (
        <Badge variant="light" size="sm">
          {row.original.family || extractModelFamily(row.original.checkpoint)}
        </Badge>
      ),
    },
    {
      accessorKey: 'model_type',
      header: 'Type',
      cell: ({ row }) => (
        <Badge variant="light" color={row.original.model_type === 'vlm' ? 'violet' : 'blue'} size="sm">
          {row.original.model_type.toUpperCase()}
        </Badge>
      ),
    },
    {
      accessorKey: 'parameters',
      header: 'Parameters',
      cell: ({ row }) => (
        <Text size="sm">
          {row.original.parameters
            ? row.original.parameters >= 1e9
              ? `${(row.original.parameters / 1e9).toFixed(1)}B`
              : `${(row.original.parameters / 1e6).toFixed(0)}M`
            : '-'}
        </Text>
      ),
    },
    {
      accessorKey: 'quantization',
      header: 'Quantization',
      cell: ({ row }) => {
        const quant = parseQuantization(row.original.checkpoint);
        return (
          <Badge variant="outline" size="sm">
            {quant || '-'}
          </Badge>
        );
      },
    },
    {
      accessorKey: 'created_at',
      header: 'Created',
      cell: ({ row }) => (
        <Text size="sm" c="dimmed">
          {formatDateTime(row.original.created_at)}
        </Text>
      ),
    },
    {
      id: 'actions',
      header: '',
      cell: ({ row }) => (
        <ActionIcon
          size="sm"
          color="red"
          variant="subtle"
          aria-label="Delete model"
          onClick={(e) => {
            e.stopPropagation();
            setSelectedModel(row.original);
            setDeleteModalOpen(true);
          }}
        >
          <IconTrash size={16} />
        </ActionIcon>
      ),
    },
  ];

  if (error) {
    return <ErrorDisplay message={error.message} fullScreen />;
  }

  return (
    <Box>
      <Group justify="space-between" mb="xl">
        <Title order={2}>Models</Title>
        <Button leftSection={<IconPlus size={18} />} onClick={() => setAddModalOpen(true)}>
          Add Model
        </Button>
      </Group>

      {/* Filters */}
      <Card padding="md" radius="md" withBorder mb="md">
        <Grid>
          <Grid.Col span={{ base: 12, md: 6, lg: 4 }}>
            <TextInput
              placeholder="Search models..."
              leftSection={<IconSearch size={16} />}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              size="sm"
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, sm: 6, lg: 3 }}>
            <Select
              placeholder="All families"
              data={families}
              value={selectedFamily}
              onChange={setSelectedFamily}
              size="sm"
              clearable
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, sm: 6, lg: 3 }}>
            <Select
              placeholder="All types"
              data={[
                { value: 'llm', label: 'LLM' },
                { value: 'vlm', label: 'VLM' },
                { value: 'embedding', label: 'Embedding' },
              ]}
              value={selectedType}
              onChange={setSelectedType}
              size="sm"
              clearable
            />
          </Grid.Col>
        </Grid>
      </Card>

      {/* Models Table */}
      <Card padding="lg" radius="md" withBorder>
        {isLoading ? (
          <Skeleton height={400} />
        ) : models.length === 0 ? (
          <Text c="dimmed" ta="center" py="xl">
            No models found. Add your first model to get started.
          </Text>
        ) : (
          <DataTable
            data={models}
            columns={columns}
            onRowClick={(row) => navigate(`/models/${row.id}`)}
          />
        )}
      </Card>

      {/* Delete Confirmation Modal */}
      <Modal
        opened={deleteModalOpen}
        onClose={() => setDeleteModalOpen(false)}
        title="Delete Model"
      >
        <Text mb="md">
          Are you sure you want to delete "{selectedModel?.name}"? This will also delete all
          associated runs and metrics.
        </Text>
        <Group justify="flex-end">
          <Button variant="outline" onClick={() => setDeleteModalOpen(false)}>
            Cancel
          </Button>
          <Button color="red" onClick={handleDelete} loading={deleteModel.isPending}>
            Delete
          </Button>
        </Group>
      </Modal>

      {/* Add Model Modal */}
      <Modal
        opened={addModalOpen}
        onClose={() => { setAddModalOpen(false); reset(); }}
        title="Add Model"
      >
        <form onSubmit={handleSubmit(handleAddModel)}>
          <Stack gap="md">
            <TextInput
              label="Name"
              placeholder="Llama 3.2 1B Instruct"
              required
              error={errors.name?.message}
              {...register('name', { required: 'Name is required' })}
            />
            <TextInput
              label="Checkpoint"
              placeholder="Llama-3.2-1B-Instruct-GGUF"
              required
              error={errors.checkpoint?.message}
              {...register('checkpoint', { required: 'Checkpoint is required' })}
            />
            <Select
              label="Type"
              data={[
                { value: 'llm', label: 'LLM' },
                { value: 'vlm', label: 'VLM (Vision-Language)' },
                { value: 'embedding', label: 'Embedding' },
              ]}
              defaultValue="llm"
              onChange={(value) => value}
              {...register('model_type')}
            />
            <TextInput
              label="HuggingFace Repo (optional)"
              placeholder="meta-llama/Llama-3.2-1B-Instruct"
              {...register('hf_repo')}
            />
            <Group justify="flex-end">
              <Button variant="outline" onClick={() => { setAddModalOpen(false); reset(); }}>
                Cancel
              </Button>
              <Button type="submit" loading={createModel.isPending}>
                Add Model
              </Button>
            </Group>
          </Stack>
        </form>
      </Modal>
    </Box>
  );
}
