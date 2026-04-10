/**
 * DataTable Component with TanStack Table
 * Reusable data table with sorting, pagination, and selection
 */

import {
  Table,
  Checkbox,
  Text,
  Group,
  ActionIcon,
  Menu,
  ScrollArea,
  Pagination,
  Select,
  Box,
} from '@mantine/core';
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getPaginationRowModel,
  getFilteredRowModel,
  flexRender,
  ColumnDef,
  SortingState,
  PaginationState,
} from '@tanstack/react-table';
import { useState, useEffect } from 'react';
import { IconDots, IconChevronUp, IconChevronDown, IconSelector } from '@tabler/icons-react';
import type { Row } from '@tanstack/react-table';

// Fix 1: RowAction interface and actions prop added to DataTableProps
interface RowAction<T> {
  label: string;
  color?: string;
  onClick: (row: T) => void;
}

interface DataTableProps<T> {
  data: T[];
  columns: ColumnDef<T>[];
  onRowClick?: (row: T) => void;
  onSelectionChange?: (selectedRows: T[]) => void;
  enableSelection?: boolean;
  enablePagination?: boolean;
  pageSize?: number;
  total?: number;
  isLoading?: boolean;
  emptyMessage?: string;
  actions?: RowAction<T>[];
}

export function DataTable<T>({
  data,
  columns,
  onRowClick,
  onSelectionChange,
  enableSelection = false,
  enablePagination = true,
  pageSize = 20,
  total,
  isLoading = false,
  emptyMessage = 'No data available',
  actions,
}: DataTableProps<T>) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [pagination, setPagination] = useState<PaginationState>({
    pageIndex: 0,
    pageSize,
  });
  const [rowSelection, setRowSelection] = useState({});

  const table = useReactTable({
    data,
    columns: [
      ...(enableSelection
        ? [
            {
              id: 'select',
              header: ({ table }: { table: typeof table }) => (
                <Checkbox
                  checked={table.getIsAllRowsSelected()}
                  indeterminate={table.getIsSomeRowsSelected()}
                  onChange={table.getToggleAllRowsSelectedHandler()}
                  aria-label="Select all rows"
                />
              ),
              cell: ({ row }: { row: Row<T> }) => (
                <Checkbox
                  checked={row.getIsSelected()}
                  onChange={row.getToggleSelectedHandler()}
                  aria-label="Select row"
                />
              ),
              enableSorting: false,
              enableHiding: false,
            } as ColumnDef<T>,
          ]
        : []),
      ...columns,
    ],
    state: {
      sorting,
      pagination,
      rowSelection,
    },
    onSortingChange: setSorting,
    onPaginationChange: setPagination,
    onRowSelectionChange: setRowSelection,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    manualPagination: total !== undefined,
    rowCount: total ?? data.length,
  });

  // Fix 3: useEffect now has dependency array so it only fires when selection changes
  useEffect(() => {
    if (onSelectionChange) {
      const selectedRows = table.getFilteredSelectedRowModel().rows.map((row) => row.original);
      onSelectionChange(selectedRows);
    }
  }, [rowSelection, onSelectionChange]);

  // Fix 2 + Fix 4: derive whether the actions column should be rendered
  const hasActions = actions !== undefined && actions.length > 0;

  return (
    <Box>
      <ScrollArea>
        <Table miw={700}>
          <Table.Thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <Table.Tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <Table.Th
                    key={header.id}
                    onClick={header.column.getToggleSortingHandler()}
                    style={{
                      cursor: header.column.getCanSort() ? 'pointer' : 'default',
                    }}
                    aria-label={header.column.getCanSort()
                      ? `Sort by ${header.column.id} ${header.column.getIsSorted() === 'asc' ? 'descending' : 'ascending'}`
                      : undefined}
                  >
                    <Group gap="xs" wrap="nowrap">
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getCanSort() && (
                        <>
                          {header.column.getIsSorted() === 'asc' && <IconChevronUp size={14} />}
                          {header.column.getIsSorted() === 'desc' && <IconChevronDown size={14} />}
                          {!header.column.getIsSorted() && <IconSelector size={14} />}
                        </>
                      )}
                    </Group>
                  </Table.Th>
                ))}
                {/* Fix 2: Only render the empty header cell when actions are provided */}
                {hasActions && <Table.Th />}
              </Table.Tr>
            ))}
          </Table.Thead>
          <Table.Tbody>
            {isLoading ? (
              <Table.Tr>
                {/* Fix 4: colSpan accounts for the conditional actions column */}
                <Table.Td
                  colSpan={
                    columns.length +
                    (enableSelection ? 1 : 0) +
                    (hasActions ? 1 : 0)
                  }
                  ta="center"
                  py="xl"
                >
                  <Text c="dimmed">Loading...</Text>
                </Table.Td>
              </Table.Tr>
            ) : table.getRowModel().rows.length === 0 ? (
              <Table.Tr>
                {/* Fix 4: colSpan accounts for the conditional actions column */}
                <Table.Td
                  colSpan={
                    columns.length +
                    (enableSelection ? 1 : 0) +
                    (hasActions ? 1 : 0)
                  }
                  ta="center"
                  py="xl"
                >
                  <Text c="dimmed">{emptyMessage}</Text>
                </Table.Td>
              </Table.Tr>
            ) : (
              table.getRowModel().rows.map((row) => (
                <Table.Tr
                  key={row.id}
                  onClick={() => onRowClick?.(row.original)}
                  style={{
                    cursor: onRowClick ? 'pointer' : 'default',
                  }}
                  bg={row.getIsSelected() ? 'var(--mantine-color-blue-light)' : undefined}
                >
                  {row.getVisibleCells().map((cell) => (
                    <Table.Td key={cell.id}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </Table.Td>
                  ))}
                  {/* Fix 2: Only render the actions cell when actions are provided */}
                  {hasActions && (
                    <Table.Td>
                      <Group gap="xs" justify="flex-end">
                        <Menu withinPortal position="bottom-end" shadow="sm">
                          <Menu.Target>
                            <ActionIcon
                              variant="subtle"
                              onClick={(e) => e.stopPropagation()}
                              aria-label="Row actions menu"
                            >
                              <IconDots size={16} />
                            </ActionIcon>
                          </Menu.Target>
                          <Menu.Dropdown>
                            {/* Fix 2: Map over actions prop instead of hardcoded items */}
                            {actions.map((a, i) => (
                              <Menu.Item
                                key={i}
                                color={a.color}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  a.onClick(row.original);
                                }}
                              >
                                {a.label}
                              </Menu.Item>
                            ))}
                          </Menu.Dropdown>
                        </Menu>
                      </Group>
                    </Table.Td>
                  )}
                </Table.Tr>
              ))
            )}
          </Table.Tbody>
        </Table>
      </ScrollArea>

      {enablePagination && (
        <Group justify="space-between" align="center" mt="md">
          <Group gap="sm">
            <Text size="sm" c="dimmed">
              Rows per page
            </Text>
            <Select
              value={table.getState().pagination.pageSize.toString()}
              onChange={(value) => table.setPageSize(Number(value))}
              data={['10', '20', '50', '100']}
              w={80}
              size="xs"
            />
          </Group>
          <Group gap="sm">
            <Text size="sm" c="dimmed">
              Page {table.getState().pagination.pageIndex + 1} of{' '}
              {table.getPageCount()}
            </Text>
            <Pagination
              value={table.getState().pagination.pageIndex + 1}
              onChange={(page) => table.setPageIndex(page - 1)}
              total={table.getPageCount()}
              size="sm"
            />
          </Group>
        </Group>
      )}
    </Box>
  );
}

export default DataTable;
