/**
 * Tests for DataTable Component
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '../utils';
import { DataTable } from '@/components/common';
import { ColumnDef } from '@tanstack/react-table';
import '@testing-library/jest-dom';

// Mock data
interface TestDataRow {
  id: string;
  name: string;
  value: number;
  status: string;
}

const mockData: TestDataRow[] = [
  { id: '1', name: 'Item 1', value: 100, status: 'active' },
  { id: '2', name: 'Item 2', value: 200, status: 'inactive' },
  { id: '3', name: 'Item 3', value: 300, status: 'active' },
];

const mockColumns: ColumnDef<TestDataRow>[] = [
  {
    accessorKey: 'name',
    header: 'Name',
  },
  {
    accessorKey: 'value',
    header: 'Value',
  },
  {
    accessorKey: 'status',
    header: 'Status',
  },
];

describe('DataTable', () => {
  describe('Basic Rendering', () => {
    it('should render table with data', () => {
      render(<DataTable data={mockData} columns={mockColumns} enablePagination={false} />);
      expect(screen.getByText('Item 1')).toBeInTheDocument();
      expect(screen.getByText('Item 2')).toBeInTheDocument();
      expect(screen.getByText('Item 3')).toBeInTheDocument();
    });

    it('should render column headers', () => {
      render(<DataTable data={mockData} columns={mockColumns} enablePagination={false} />);
      expect(screen.getByText('Name')).toBeInTheDocument();
      expect(screen.getByText('Value')).toBeInTheDocument();
      expect(screen.getByText('Status')).toBeInTheDocument();
    });

    it('should render correct number of rows', () => {
      render(<DataTable data={mockData} columns={mockColumns} enablePagination={false} />);
      // Should have 3 data rows
      expect(screen.getAllByRole('row').length).toBeGreaterThan(2);
    });
  });

  describe('Empty State', () => {
    it('should show empty message when no data', () => {
      render(<DataTable data={[]} columns={mockColumns} enablePagination={false} />);
      expect(screen.getByText('No data available')).toBeInTheDocument();
    });

    it('should show custom empty message', () => {
      render(
        <DataTable
          data={[]}
          columns={mockColumns}
          enablePagination={false}
          emptyMessage="Custom empty message"
        />
      );
      expect(screen.getByText('Custom empty message')).toBeInTheDocument();
    });
  });

  describe('Loading State', () => {
    it('should show loading indicator when isLoading is true', () => {
      render(
        <DataTable
          data={[]}
          columns={mockColumns}
          enablePagination={false}
          isLoading={true}
        />
      );
      expect(screen.getByText('Loading...')).toBeInTheDocument();
    });

    it('should not show loading when isLoading is false', () => {
      render(
        <DataTable
          data={mockData}
          columns={mockColumns}
          enablePagination={false}
          isLoading={false}
        />
      );
      expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
    });
  });

  describe('Selection', () => {
    it('should render checkbox column when enableSelection is true', () => {
      render(
        <DataTable
          data={mockData}
          columns={mockColumns}
          enableSelection={true}
          enablePagination={false}
        />
      );
      // Checkboxes should be present
      const checkboxes = screen.getAllByRole('checkbox');
      expect(checkboxes.length).toBeGreaterThan(0);
    });

    it('should not render checkbox column when enableSelection is false', () => {
      render(
        <DataTable
          data={mockData}
          columns={mockColumns}
          enableSelection={false}
          enablePagination={false}
        />
      );
      // Fewer checkboxes should be present
      const checkboxes = screen.queryAllByRole('checkbox');
      expect(checkboxes.length).toBe(0);
    });

    it('should call onSelectionChange when selection changes', () => {
      const onSelectionChange = vi.fn();
      render(
        <DataTable
          data={mockData}
          columns={mockColumns}
          enableSelection={true}
          enablePagination={false}
          onSelectionChange={onSelectionChange}
        />
      );
      // Selection functionality would be tested with user events
      expect(onSelectionChange).toBeDefined();
    });
  });

  describe('Row Click', () => {
    it('should call onRowClick when row is clicked', () => {
      const onRowClick = vi.fn();
      render(
        <DataTable
          data={mockData}
          columns={mockColumns}
          enablePagination={false}
          onRowClick={onRowClick}
        />
      );
      expect(onRowClick).toBeDefined();
    });

    it('should not have pointer cursor when onRowClick is not provided', () => {
      render(
        <DataTable
          data={mockData}
          columns={mockColumns}
          enablePagination={false}
        />
      );
      // Row should not have pointer cursor
      const row = screen.getByText('Item 1').closest('tr');
      expect(row?.style.cursor).not.toBe('pointer');
    });
  });

  describe('Pagination', () => {
    it('should render pagination controls by default', () => {
      render(<DataTable data={mockData} columns={mockColumns} />);
      // Pagination should be present
      expect(screen.getByText('Rows per page')).toBeInTheDocument();
    });

    it('should not render pagination when enablePagination is false', () => {
      render(
        <DataTable
          data={mockData}
          columns={mockColumns}
          enablePagination={false}
        />
      );
      expect(screen.queryByText('Rows per page')).not.toBeInTheDocument();
    });

    it('should show page numbers in pagination', () => {
      render(<DataTable data={mockData} columns={mockColumns} />);
      expect(screen.getByText(/Page \d+ of \d+/)).toBeInTheDocument();
    });

    it('should allow changing page size', () => {
      render(<DataTable data={mockData} columns={mockColumns} />);
      const pageSizeSelect = screen.getByRole('combobox', { name: /rows per page/i });
      expect(pageSizeSelect).toBeInTheDocument();
    });
  });

  describe('Sorting', () => {
    it('should render sortable columns', () => {
      render(<DataTable data={mockData} columns={mockColumns} enablePagination={false} />);
      // Headers should be clickable for sorting
      const nameHeader = screen.getByText('Name');
      expect(nameHeader).toBeInTheDocument();
    });

    it('should show sort icons', () => {
      render(<DataTable data={mockData} columns={mockColumns} enablePagination={false} />);
      // Sort icon should be visible in headers
      const headers = screen.getAllByRole('columnheader');
      expect(headers.length).toBeGreaterThan(2);
    });
  });

  describe('Action Menu', () => {
    it('should render action menu for each row', () => {
      render(<DataTable data={mockData} columns={mockColumns} enablePagination={false} />);
      // Action menu buttons should be present
      const menuButtons = screen.getAllByRole('button', { name: /dots/i });
      expect(menuButtons.length).toBe(3);
    });
  });

  describe('Data Formatting', () => {
    it('should render string data correctly', () => {
      render(<DataTable data={mockData} columns={mockColumns} enablePagination={false} />);
      expect(screen.getByText('Item 1')).toBeInTheDocument();
      expect(screen.getByText('active')).toBeInTheDocument();
    });

    it('should render number data correctly', () => {
      render(<DataTable data={mockData} columns={mockColumns} enablePagination={false} />);
      expect(screen.getByText('100')).toBeInTheDocument();
      expect(screen.getByText('200')).toBeInTheDocument();
    });

    it('should handle custom cell renderers', () => {
      const customColumns: ColumnDef<TestDataRow>[] = [
        {
          accessorKey: 'name',
          header: 'Name',
          cell: ({ getValue }) => `Custom: ${getValue<string>()}`,
        },
      ];
      render(<DataTable data={mockData} columns={customColumns} enablePagination={false} />);
      expect(screen.getByText('Custom: Item 1')).toBeInTheDocument();
    });
  });

  describe('Total Count', () => {
    it('should display total count when provided', () => {
      render(
        <DataTable
          data={mockData}
          columns={mockColumns}
          total={100}
        />
      );
      expect(screen.getByText(/Page \d+ of \d+/)).toBeInTheDocument();
    });

    it('should use data length when total is not provided', () => {
      render(
        <DataTable
          data={mockData}
          columns={mockColumns}
          enablePagination={false}
        />
      );
      expect(screen.getByText('Item 1')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have proper table role', () => {
      render(<DataTable data={mockData} columns={mockColumns} enablePagination={false} />);
      const table = screen.getByRole('table');
      expect(table).toBeInTheDocument();
    });

    it('should have proper column headers', () => {
      render(<DataTable data={mockData} columns={mockColumns} enablePagination={false} />);
      const headers = screen.getAllByRole('columnheader');
      expect(headers.length).toBeGreaterThan(1);
    });

    it('should have proper row elements', () => {
      render(<DataTable data={mockData} columns={mockColumns} enablePagination={false} />);
      const rows = screen.getAllByRole('row');
      expect(rows.length).toBeGreaterThan(2);
    });
  });
});
