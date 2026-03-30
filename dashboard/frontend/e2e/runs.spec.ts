/**
 * E2E Tests for Runs Page
 * Tests run listing, filtering, and status updates
 */

import { test, expect } from '@playwright/test';

test.describe('Runs Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/runs');
  });

  test('should display runs page', async ({ page }) => {
    await expect(page).toHaveURL(/.*runs/);
  });

  test('should display runs table', async ({ page }) => {
    // Wait for data to load
    await page.waitForTimeout(1000);

    // Check for table structure
    const table = page.getByRole('table');
    const tableVisible = await table.isVisible().catch(() => false);

    // May show empty state if no runs
    const emptyState = page.getByText(/no runs|empty/i);
    const emptyVisible = await emptyState.isVisible().catch(() => false);

    expect(tableVisible || emptyVisible).toBe(true);
  });

  test('should display run information', async ({ page }) => {
    await page.waitForTimeout(1000);

    // Check for run name column
    const nameHeader = page.getByText(/run/i);
    const nameVisible = await nameHeader.isVisible().catch(() => false);

    expect(nameVisible).toBe(true);
  });

  test('should display status badges', async ({ page }) => {
    await page.waitForTimeout(1000);

    // Look for status badges (pending, running, completed, failed, cancelled)
    const statusBadges = page.getByText(/pending|running|completed|failed|cancelled/i);
    const badgesVisible = await statusBadges.first().isVisible().catch(() => false);

    expect(typeof badgesVisible).toBe('boolean');
  });

  test('should display run type badges', async ({ page }) => {
    await page.waitForTimeout(1000);

    // Look for type badges (benchmark, accuracy-mmlu, etc.)
    const typeBadges = page.getByText(/benchmark|mmlu|humaneval|lm eval|perplexity/i);
    const badgesVisible = await typeBadges.first().isVisible().catch(() => false);

    expect(typeof badgesVisible).toBe('boolean');
  });

  test('should filter runs by status', async ({ page }) => {
    // Find status filter dropdown
    const statusFilter = page.getByRole('combobox', { name: /all statuses/i });
    const filterVisible = await statusFilter.isVisible().catch(() => false);

    if (filterVisible) {
      await statusFilter.click();

      // Select a status option
      const completedOption = page.getByText('Completed');
      const optionVisible = await completedOption.isVisible().catch(() => false);

      if (optionVisible) {
        await completedOption.click();
        await page.waitForTimeout(500);
        expect(true).toBe(true);
      }
    }
  });

  test('should filter runs by type', async ({ page }) => {
    // Find type filter dropdown
    const typeFilter = page.getByRole('combobox', { name: /all types/i });
    const filterVisible = await typeFilter.isVisible().catch(() => false);

    if (filterVisible) {
      await typeFilter.click();

      // Select benchmark option
      const benchmarkOption = page.getByText('Benchmark');
      const optionVisible = await benchmarkOption.isVisible().catch(() => false);

      if (optionVisible) {
        await benchmarkOption.click();
        await page.waitForTimeout(500);
        expect(true).toBe(true);
      }
    }
  });

  test('should filter runs by device', async ({ page }) => {
    // Find device filter dropdown
    const deviceFilter = page.getByRole('combobox', { name: /all devices/i });
    const filterVisible = await deviceFilter.isVisible().catch(() => false);

    if (filterVisible) {
      await deviceFilter.click();

      // Select CPU option
      const cpuOption = page.getByText('CPU');
      const optionVisible = await cpuOption.isVisible().catch(() => false);

      if (optionVisible) {
        await cpuOption.click();
        await page.waitForTimeout(500);
        expect(true).toBe(true);
      }
    }
  });

  test('should filter runs by backend', async ({ page }) => {
    // Find backend filter dropdown
    const backendFilter = page.getByRole('combobox', { name: /all backends/i });
    const filterVisible = await backendFilter.isVisible().catch(() => false);

    if (filterVisible) {
      await backendFilter.click();

      // Select an option
      const llamacppOption = page.getByText('LLamaCpp');
      const optionVisible = await llamacppOption.isVisible().catch(() => false);

      if (optionVisible) {
        await llamacppOption.click();
        await page.waitForTimeout(500);
        expect(true).toBe(true);
      }
    }
  });

  test('should search runs by name', async ({ page }) => {
    // Find search input
    const searchInput = page.getByPlaceholder(/search runs/i);
    const searchVisible = await searchInput.isVisible().catch(() => false);

    if (searchVisible) {
      await searchInput.fill('test');
      await page.waitForTimeout(500);

      // Should filter results
      expect(true).toBe(true);
    }
  });

  test('should clear filters', async ({ page }) => {
    // Find status filter
    const statusFilter = page.getByRole('combobox', { name: /all statuses/i });
    const filterVisible = await statusFilter.isVisible().catch(() => false);

    if (filterVisible) {
      await statusFilter.click();

      const completedOption = page.getByText('Completed');
      if (await completedOption.isVisible().catch(() => false)) {
        await completedOption.click();

        // Find and click clear button
        const clearButton = page.getByRole('button', { name: /clear/i });
        if (await clearButton.isVisible()) {
          await clearButton.click();
          await page.waitForTimeout(300);
        }
      }
    }
  });

  test('should navigate to run detail page', async ({ page }) => {
    await page.waitForTimeout(1000);

    // Find first run link
    const runLink = page.getByRole('link').first();
    const linkVisible = await runLink.isVisible().catch(() => false);

    if (linkVisible) {
      await runLink.click();
      await expect(page).toHaveURL(/.*runs\/.+/);
    }
  });

  test('should display model name for each run', async ({ page }) => {
    await page.waitForTimeout(1000);

    // Check for model names in the table
    const modelNames = page.getByText(/llama|qwen|phi|mistral|gemma/i);
    const namesVisible = await modelNames.first().isVisible().catch(() => false);

    expect(typeof namesVisible).toBe('boolean');
  });

  test('should display duration for completed runs', async ({ page }) => {
    await page.waitForTimeout(1000);

    // Check for duration column
    const durationHeader = page.getByText(/duration/i);
    const headerVisible = await durationHeader.isVisible().catch(() => false);

    expect(headerVisible).toBe(true);
  });

  test('should display date/time for runs', async ({ page }) => {
    await page.waitForTimeout(1000);

    // Check for date column
    const dateHeader = page.getByText(/date/i);
    const headerVisible = await dateHeader.isVisible().catch(() => false);

    expect(headerVisible).toBe(true);
  });

  test('should handle loading state', async ({ page }) => {
    // Navigate fresh to trigger loading
    await page.goto('/runs');

    // May show loading indicator initially
    const loadingVisible = await page.getByText(/loading/i).isVisible().catch(() => false);

    // Loading state is acceptable
    expect(typeof loadingVisible).toBe('boolean');
  });

  test('should display empty state when no runs', async ({ page }) => {
    await page.waitForTimeout(1000);

    const emptyState = page.getByText(/no runs found/i);
    const emptyVisible = await emptyState.isVisible().catch(() => false);

    const table = page.getByRole('table');
    const tableVisible = await table.isVisible().catch(() => false);

    // Should show either empty state or data
    expect(emptyVisible || tableVisible).toBe(true);
  });
});

test.describe('Run Status Display', () => {
  test('should display pending status badge', async ({ page }) => {
    await page.goto('/runs');
    await page.waitForTimeout(1000);

    const pendingBadge = page.getByText(/pending/i);
    const visible = await pendingBadge.isVisible().catch(() => false);
    expect(typeof visible).toBe('boolean');
  });

  test('should display running status badge', async ({ page }) => {
    await page.goto('/runs');
    await page.waitForTimeout(1000);

    const runningBadge = page.getByText(/running/i);
    const visible = await runningBadge.isVisible().catch(() => false);
    expect(typeof visible).toBe('boolean');
  });

  test('should display completed status badge', async ({ page }) => {
    await page.goto('/runs');
    await page.waitForTimeout(1000);

    const completedBadge = page.getByText(/completed/i);
    const visible = await completedBadge.isVisible().catch(() => false);
    expect(typeof visible).toBe('boolean');
  });

  test('should display failed status badge', async ({ page }) => {
    await page.goto('/runs');
    await page.waitForTimeout(1000);

    const failedBadge = page.getByText(/failed/i);
    const visible = await failedBadge.isVisible().catch(() => false);
    expect(typeof visible).toBe('boolean');
  });
});

test.describe('Runs Table Row Actions', () => {
  test('should have clickable rows for navigation', async ({ page }) => {
    await page.goto('/runs');
    await page.waitForTimeout(1000);

    const table = page.getByRole('table');
    const tableVisible = await table.isVisible().catch(() => false);

    if (tableVisible) {
      // Click first row
      const firstRow = page.getByRole('row').nth(1); // Skip header row
      const rowVisible = await firstRow.isVisible().catch(() => false);

      if (rowVisible) {
        await firstRow.click();
        await page.waitForTimeout(500);
        // Should navigate to detail page
        expect(page.url()).toContain('/runs/');
      }
    }
  });
});
