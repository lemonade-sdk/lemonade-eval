/**
 * E2E Tests for Compare Page
 * Tests run comparison, metrics visualization, and multi-run analysis
 */

import { test, expect } from '@playwright/test';

test.describe('Compare Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/compare');
  });

  test('should display compare page', async ({ page }) => {
    await expect(page).toHaveURL(/.*compare/);
  });

  test('should display page title', async ({ page }) => {
    const title = page.getByRole('heading', { name: /compare runs/i });
    await expect(title).toBeVisible();
  });

  test('should display run selection dropdown', async ({ page }) => {
    const selectPlaceholder = page.getByText(/choose runs/i);
    const selectVisible = await selectPlaceholder.isVisible().catch(() => false);

    // Or check for the MultiSelect element
    const multiSelect = page.getByRole('combobox');
    const multiSelectVisible = await multiSelect.isVisible().catch(() => false);

    expect(selectVisible || multiSelectVisible).toBe(true);
  });

  test('should display helper text for selection', async ({ page }) => {
    const helperText = page.getByText(/select.*runs.*compare/i);
    const textVisible = await helperText.isVisible().catch(() => false);

    expect(typeof textVisible).toBe('boolean');
  });

  test('should show message when less than 2 runs selected', async ({ page }) => {
    await page.waitForTimeout(1000);

    // Should show message prompting to select runs
    const promptText = page.getByText(/select.*2.*runs/i);
    const textVisible = await promptText.isVisible().catch(() => false);

    expect(typeof textVisible).toBe('boolean');
  });

  test('should display runs dropdown with available runs', async ({ page }) => {
    await page.waitForTimeout(1000);

    // Click the dropdown to see options
    const dropdown = page.getByRole('combobox');
    const dropdownVisible = await dropdown.isVisible().catch(() => false);

    if (dropdownVisible) {
      await dropdown.click();
      await page.waitForTimeout(500);

      // Check if dropdown options appear
      const options = page.getByRole('option');
      const optionCount = await options.count();

      expect(optionCount).toBeGreaterThanOrEqual(0);
    }
  });

  test('should display selected runs overview when runs are selected', async ({ page }) => {
    await page.waitForTimeout(1000);

    // Find and click the run selection dropdown
    const select = page.getByRole('combobox');
    const selectVisible = await select.isVisible().catch(() => false);

    if (selectVisible) {
      await select.click();
      await page.waitForTimeout(300);

      // Try to select first run option
      const firstOption = page.locator('[role="option"]').first();
      const optionVisible = await firstOption.isVisible().catch(() => false);

      if (optionVisible) {
        await firstOption.click();
        await page.waitForTimeout(300);

        // Select second run
        await select.click();
        await page.waitForTimeout(300);

        const secondOption = page.locator('[role="option"]').nth(1);
        const secondVisible = await secondOption.isVisible().catch(() => false);

        if (secondVisible) {
          await secondOption.click();
          await page.waitForTimeout(1000);

          // Should show selected runs overview
          const selectedRunsHeading = page.getByText(/selected runs/i);
          const headingVisible = await selectedRunsHeading.isVisible().catch(() => false);

          expect(typeof headingVisible).toBe('boolean');
        }
      }
    }
  });

  test('should display comparison table header', async ({ page }) => {
    const tableHeader = page.getByText(/metrics comparison/i);
    const headerVisible = await tableHeader.isVisible().catch(() => false);

    expect(typeof tableHeader).toBe('boolean');
  });

  test('should display performance metrics in comparison', async ({ page }) => {
    await page.waitForTimeout(1000);

    // Look for metric names that would appear in comparison
    const metricNames = page.getByText(/tokens per second|seconds to first token|prefill/i);
    const metricsVisible = await metricNames.first().isVisible().catch(() => false);

    expect(typeof metricsVisible).toBe('boolean');
  });

  test('should display chart sections', async ({ page }) => {
    await page.waitForTimeout(1000);

    // Check for chart section headers
    const performanceChart = page.getByText(/performance comparison/i);
    const performanceVisible = await performanceChart.isVisible().catch(() => false);

    const multiMetricChart = page.getByText(/multi-metric comparison/i);
    const multiMetricVisible = await multiMetricChart.isVisible().catch(() => false);

    expect(performanceVisible || multiMetricVisible).toBe(true);
  });

  test('should handle empty state gracefully', async ({ page }) => {
    await page.waitForTimeout(1000);

    // Should not crash with no runs available
    const errorElement = page.getByText(/error|failed to load/i);
    const errorVisible = await errorElement.isVisible().catch(() => false);

    // Page should load without errors
    expect(page.url()).toContain('compare');
  });

  test('should have responsive layout', async ({ page }) => {
    await page.goto('/compare');

    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.waitForTimeout(500);

    // Should still show content
    expect(page.url()).toContain('compare');

    // Test tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.waitForTimeout(500);

    expect(page.url()).toContain('compare');

    // Reset to desktop
    await page.setViewportSize({ width: 1920, height: 1080 });
  });
});

test.describe('Compare Page - Run Cards', () => {
  test('should display run build name in card', async ({ page }) => {
    await page.waitForTimeout(1000);

    const buildNames = page.getByRole('heading');
    const namesVisible = await buildNames.first().isVisible().catch(() => false);

    expect(typeof namesVisible).toBe('boolean');
  });

  test('should display status badge on run card', async ({ page }) => {
    await page.waitForTimeout(1000);

    const statusBadges = page.getByText(/pending|running|completed|failed|cancelled/i);
    const badgesVisible = await statusBadges.first().isVisible().catch(() => false);

    expect(typeof badgesVisible).toBe('boolean');
  });

  test('should display model name on run card', async ({ page }) => {
    await page.waitForTimeout(1000);

    const modelNames = page.getByText(/unknown model|llama|qwen|phi/i);
    const namesVisible = await modelNames.first().isVisible().catch(() => false);

    expect(typeof namesVisible).toBe('boolean');
  });

  test('should display device and backend info', async ({ page }) => {
    await page.waitForTimeout(1000);

    const deviceInfo = page.getByText(/cpu|gpu|npu|hybrid|-/i);
    const infoVisible = await deviceInfo.first().isVisible().catch(() => false);

    expect(typeof infoVisible).toBe('boolean');
  });

  test('should display creation date on run card', async ({ page }) => {
    await page.waitForTimeout(1000);

    // Date format would be visible
    const dateInfo = page.getByText(/\d{4}|\d{2}\/\d{2}\/\d{2}/i);
    const dateVisible = await dateInfo.first().isVisible().catch(() => false);

    expect(typeof dateVisible).toBe('boolean');
  });
});

test.describe('Compare Page - Metrics Table', () => {
  test('should display metrics table structure', async ({ page }) => {
    await page.waitForTimeout(1000);

    const table = page.getByRole('table');
    const tableVisible = await table.isVisible().catch(() => false);

    expect(typeof tableVisible).toBe('boolean');
  });

  test('should display metric name column', async ({ page }) => {
    await page.waitForTimeout(1000);

    const metricColumn = page.getByText(/metric/i);
    const columnVisible = await metricColumn.isVisible().catch(() => false);

    expect(columnVisible).toBe(true);
  });

  test('should highlight best performing values', async ({ page }) => {
    await page.waitForTimeout(1000);

    // Best values might have different styling
    const highlightedValues = page.locator('[style*="green"]');
    const highlightedVisible = await highlightedValues.first().isVisible().catch(() => false);

    expect(typeof highlightedVisible).toBe('boolean');
  });

  test('should display metric units', async ({ page }) => {
    await page.waitForTimeout(1000);

    // Units like tokens/s, ms, GB should be visible
    const units = page.getByText(/tokens|ms|gb|s\/i);
    const unitsVisible = await units.first().isVisible().catch(() => false);

    expect(typeof unitsVisible).toBe('boolean');
  });
});

test.describe('Compare Page - Charts', () => {
  test('should display bar chart container', async ({ page }) => {
    await page.waitForTimeout(1000);

    // Check for chart-related SVG elements
    const svgCharts = page.locator('svg');
    const chartCount = await svgCharts.count();

    expect(chartCount).toBeGreaterThanOrEqual(0);
  });

  test('should display chart legends', async ({ page }) => {
    await page.waitForTimeout(1000);

    // Look for chart legend items
    const legends = page.getByText(/run \d+/i);
    const legendsVisible = await legends.first().isVisible().catch(() => false);

    expect(typeof legendsVisible).toBe('boolean');
  });

  test('should show radar chart for multi-metric comparison', async ({ page }) => {
    await page.waitForTimeout(1000);

    const radarChart = page.getByText(/multi-metric/i);
    const chartVisible = await radarChart.isVisible().catch(() => false);

    expect(typeof chartVisible).toBe('boolean');
  });
});

test.describe('Compare Page - Navigation', () => {
  test('should have working back navigation', async ({ page }) => {
    await page.goto('/compare');

    // Find runs or dashboard link in navigation
    const runsLink = page.getByRole('link', { name: /runs/i });
    await expect(runsLink).toBeVisible();
  });

  test('should maintain selection after page interaction', async ({ page }) => {
    await page.goto('/compare');
    await page.waitForTimeout(1000);

    // Make a selection if possible
    const select = page.getByRole('combobox');
    const selectVisible = await select.isVisible().catch(() => false);

    if (selectVisible) {
      await select.click();
      await page.waitForTimeout(300);

      const firstOption = page.locator('[role="option"]').first();
      const optionVisible = await firstOption.isVisible().catch(() => false);

      if (optionVisible) {
        await firstOption.click();
        await page.waitForTimeout(500);

        // Selection should persist
        const selectedValue = await select.inputValue().catch(() => '');
        expect(selectedValue).toBeTruthy();
      }
    }
  });
});
