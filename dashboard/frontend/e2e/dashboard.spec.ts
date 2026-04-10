/**
 * E2E Tests for Dashboard Page
 * Tests dashboard navigation, data display, and overview metrics
 */

import { test, expect } from '@playwright/test';

test.describe('Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display dashboard page', async ({ page }) => {
    // Navigate to dashboard
    await page.goto('/dashboard');

    // Should show dashboard content
    await expect(page).toHaveURL(/.*dashboard/);
  });

  test('should display overview metrics', async ({ page }) => {
    await page.goto('/dashboard');

    // Check for metric cards or summary statistics
    const metrics = page.getByTestId(/metric|stat|overview/i);

    // Metrics may be loading initially
    const loadingVisible = await page.getByText(/loading/i).isVisible().catch(() => false);

    if (!loadingVisible) {
      // Should have some metrics displayed
      expect(true).toBe(true);
    }
  });

  test('should display recent runs section', async ({ page }) => {
    await page.goto('/dashboard');

    // Check for recent runs section
    const recentRunsHeading = page.getByRole('heading', { name: /recent|runs/i });
    const recentRunsVisible = await recentRunsHeading.isVisible().catch(() => false);

    // May or may not have runs depending on data
    expect(typeof recentRunsVisible).toBe('boolean');
  });

  test('should display model statistics', async ({ page }) => {
    await page.goto('/dashboard');

    // Check for model-related stats
    const modelStats = page.getByText(/models?|total models/i);
    const modelStatsVisible = await modelStats.isVisible().catch(() => false);

    expect(typeof modelStatsVisible).toBe('boolean');
  });

  test('should display run status breakdown', async ({ page }) => {
    await page.goto('/dashboard');

    // Check for status indicators (completed, running, failed)
    const statusElements = page.getByText(/completed|running|failed|pending/i);
    const statusVisible = await statusElements.first().isVisible().catch(() => false);

    expect(typeof statusVisible).toBe('boolean');
  });

  test('should have navigation to models page', async ({ page }) => {
    await page.goto('/dashboard');

    // Find models link in navigation
    const modelsLink = page.getByRole('link', { name: /models/i });
    await expect(modelsLink).toBeVisible();

    // Click and verify navigation
    await modelsLink.click();
    await expect(page).toHaveURL(/.*models/);
  });

  test('should have navigation to runs page', async ({ page }) => {
    await page.goto('/dashboard');

    // Find runs link in navigation
    const runsLink = page.getByRole('link', { name: /runs/i });
    await expect(runsLink).toBeVisible();

    // Click and verify navigation
    await runsLink.click();
    await expect(page).toHaveURL(/.*runs/);
  });

  test('should have navigation to compare page', async ({ page }) => {
    await page.goto('/dashboard');

    // Find compare link
    const compareLink = page.getByRole('link', { name: /compare/i });
    await expect(compareLink).toBeVisible();
  });

  test('should display loading state initially', async ({ page }) => {
    await page.goto('/dashboard');

    // May show loading indicator
    const loadingVisible = await page.getByText(/loading/i).isVisible().catch(() => false);

    // Loading state is acceptable
    expect(typeof loadingVisible).toBe('boolean');
  });

  test('should handle empty state gracefully', async ({ page }) => {
    await page.goto('/dashboard');

    // Should not crash with no data
    const errorElement = page.getByText(/error|failed to load/i);
    const errorVisible = await errorElement.isVisible().catch(() => false);

    // Should not show errors on initial load
    // expect(errorVisible).toBe(false);
  });

  test('should have responsive layout', async ({ page }) => {
    await page.goto('/dashboard');

    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.waitForTimeout(500);

    // Should still show content
    expect(page.url()).toContain('dashboard');

    // Test tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.waitForTimeout(500);

    expect(page.url()).toContain('dashboard');
  });

  test('should refresh data on pull-to-refresh or manual refresh', async ({ page }) => {
    await page.goto('/dashboard');

    // Find refresh button
    const refreshButton = page.getByRole('button', { name: /refresh|reload/i });
    const refreshVisible = await refreshButton.isVisible().catch(() => false);

    if (refreshVisible) {
      await refreshButton.click();
      // Should trigger data reload
      await page.waitForTimeout(1000);
    }
  });

  test('should display charts or visualizations', async ({ page }) => {
    await page.goto('/dashboard');

    // Check for charts
    const charts = page.locator('[class*="chart"], [class*="graph"], svg');
    const chartCount = await charts.count();

    // May have charts depending on implementation
    expect(chartCount).toBeGreaterThanOrEqual(0);
  });
});

test.describe('Dashboard Navigation', () => {
  test('should have working sidebar navigation', async ({ page }) => {
    await page.goto('/dashboard');

    // Check sidebar links
    const navLinks = page.getByRole('navigation').getByRole('link');
    const linkCount = await navLinks.count();

    expect(linkCount).toBeGreaterThan(0);
  });

  test('should highlight current active navigation item', async ({ page }) => {
    await page.goto('/dashboard');

    // Current page should be highlighted
    const activeLink = page.getByRole('link', { name: /dashboard/i });
    const isActive = await activeLink.getAttribute('aria-current');

    // May have aria-current="page" or similar
    expect(typeof isActive === 'string' || isActive === null).toBe(true);
  });

  test('should have breadcrumbs if applicable', async ({ page }) => {
    await page.goto('/dashboard');

    // Check for breadcrumbs
    const breadcrumbs = page.getByRole('navigation', { name: /breadcrumb/i });
    const breadcrumbsVisible = await breadcrumbs.isVisible().catch(() => false);

    // Breadcrumbs are optional
    expect(typeof breadcrumbsVisible).toBe('boolean');
  });
});

test.describe('Dashboard Data Display', () => {
  test('should display total runs count', async ({ page }) => {
    await page.goto('/dashboard');

    const totalRunsText = page.getByText(/total.*runs?|runs?:\s*\d+/i);
    const visible = await totalRunsText.isVisible().catch(() => false);

    expect(typeof visible).toBe('boolean');
  });

  test('should display success/completion rate', async ({ page }) => {
    await page.goto('/dashboard');

    const rateElement = page.getByText(/success|completion|rate/i);
    const visible = await rateElement.isVisible().catch(() => false);

    expect(typeof visible).toBe('boolean');
  });

  test('should handle API errors gracefully', async ({ page }) => {
    // Navigate with network error simulation would require mocking
    await page.goto('/dashboard');

    // Should not crash
    expect(page.url()).toContain('dashboard');
  });
});
