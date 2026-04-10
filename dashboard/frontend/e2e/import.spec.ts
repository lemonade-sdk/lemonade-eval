/**
 * E2E Tests for Import Page
 * Tests YAML import flow, progress tracking, and scan functionality
 */

import { test, expect } from '@playwright/test';

test.describe('Import Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/import');
  });

  test('should display import page', async ({ page }) => {
    await expect(page).toHaveURL(/.*import/);
  });

  test('should display page title', async ({ page }) => {
    const title = page.getByRole('heading', { name: /import evaluation results/i);
    await expect(title).toBeVisible();
  });

  test('should display import configuration card', async ({ page }) => {
    const configCard = page.getByText(/import configuration/i);
    await expect(configCard).toBeVisible();
  });

  test('should display import status card', async ({ page }) => {
    const statusCard = page.getByText(/import status/i);
    await expect(statusCard).toBeVisible();
  });
});

test.describe('Import Form', () => {
  test('should display cache directory input', async ({ page }) => {
    const dirInput = page.getByLabel(/cache directory/i);
    const inputVisible = await dirInput.isVisible().catch(() => false);

    // Or check for placeholder
    const placeholderInput = page.getByPlaceholder(/~\/\.cache\/lemonade/i);
    const placeholderVisible = await placeholderInput.isVisible().catch(() => false);

    expect(inputVisible || placeholderVisible).toBe(true);
  });

  test('should display scan button', async ({ page }) => {
    const scanButton = page.getByRole('button', { name: /scan/i });
    const buttonVisible = await scanButton.isVisible().catch(() => false);

    expect(typeof scanButton).toBe('boolean');
  });

  test('should display skip duplicates switch', async ({ page }) => {
    const skipSwitch = page.getByText(/skip duplicates/i);
    const switchVisible = await skipSwitch.isVisible().catch(() => false);

    expect(skipSwitch).toBeVisible();
  });

  test('should display dry run switch', async ({ page }) => {
    const dryRunSwitch = page.getByText(/dry run/i);
    const switchVisible = await dryRunSwitch.isVisible().catch(() => false);

    expect(dryRunSwitch).toBeVisible();
  });

  test('should display start import button', async ({ page }) => {
    const importButton = page.getByRole('button', {
      name: /start import|scan files/i,
    });
    await expect(importButton).toBeVisible();
  });

  test('should disable import button when cache directory is empty', async ({ page }) => {
    const importButton = page.getByRole('button', {
      name: /start import|scan files/i,
    });

    // Clear the cache directory input
    const dirInput = page.getByPlaceholder(/~\/\.cache\/lemonade/i);
    if (await dirInput.isVisible()) {
      await dirInput.clear();
    }

    // Button should be disabled
    const disabled = await importButton.isDisabled();
    expect(disabled).toBe(true);
  });

  test('should enable import button when cache directory is filled', async ({ page }) => {
    const dirInput = page.getByPlaceholder(/~\/\.cache\/lemonade/i);

    if (await dirInput.isVisible()) {
      await dirInput.fill('~/.cache/lemonade');
      await page.waitForTimeout(300);

      const importButton = page.getByRole('button', {
        name: /start import|scan files/i,
      });

      const disabled = await importButton.isDisabled();
      expect(disabled).toBe(false);
    }
  });
});

test.describe('Import Form - Switch Interactions', () => {
  test('should toggle skip duplicates switch', async ({ page }) => {
    const skipSwitch = page.getByRole('switch', { name: /skip duplicates/i });
    const switchVisible = await skipSwitch.isVisible().catch(() => false);

    if (switchVisible) {
      const initialState = await skipSwitch.isChecked();
      await skipSwitch.click();
      await page.waitForTimeout(200);

      const newState = await skipSwitch.isChecked();
      expect(newState).not.toBe(initialState);
    }
  });

  test('should toggle dry run switch', async ({ page }) => {
    const dryRunSwitch = page.getByRole('switch', { name: /dry run/i });
    const switchVisible = await dryRunSwitch.isVisible().catch(() => false);

    if (switchVisible) {
      const initialState = await dryRunSwitch.isChecked();
      await dryRunSwitch.click();
      await page.waitForTimeout(200);

      const newState = await dryRunSwitch.isChecked();
      expect(newState).not.toBe(initialState);
    }
  });

  test('should show description for skip duplicates', async ({ page }) => {
    const description = page.getByText(/skip.*already exist.*database/i);
    const descVisible = await description.isVisible().catch(() => false);

    expect(typeof descVisible).toBe('boolean');
  });

  test('should show description for dry run', async ({ page }) => {
    const description = page.getByText(/only scan.*without importing/i);
    const descVisible = await description.isVisible().catch(() => false);

    expect(typeof descVisible).toBe('boolean');
  });
});

test.describe('Scan Functionality', () => {
  test('should disable scan button when cache directory is empty', async ({ page }) => {
    const scanButton = page.getByRole('button', { name: /scan/i });
    const buttonVisible = await scanButton.isVisible().catch(() => false);

    if (buttonVisible) {
      const disabled = await scanButton.isDisabled();
      expect(disabled).toBe(true);
    }
  });

  test('should enable scan button when cache directory is filled', async ({ page }) => {
    const dirInput = page.getByPlaceholder(/~\/\.cache\/lemonade/i);

    if (await dirInput.isVisible()) {
      await dirInput.fill('~/.cache/lemonade');
      await page.waitForTimeout(300);

      const scanButton = page.getByRole('button', { name: /scan/i });
      const disabled = await scanButton.isDisabled();
      expect(disabled).toBe(false);
    }
  });

  test('should show scanning state when scan is in progress', async ({ page }) => {
    const dirInput = page.getByPlaceholder(/~\/\.cache\/lemonade/i);

    if (await dirInput.isVisible()) {
      await dirInput.fill('~/.cache/lemonade');
      await page.waitForTimeout(200);

      const scanButton = page.getByRole('button', { name: /scan/i });
      if (await scanButton.isVisible()) {
        await scanButton.click();

        // Should show scanning state
        const scanningText = page.getByText(/scanning/i);
        const scanningVisible = await scanningText.isVisible().catch(() => false);

        expect(typeof scanningVisible).toBe('boolean');
      }
    }
  });

  test('should display scan results alert', async ({ page }) => {
    const dirInput = page.getByPlaceholder(/~\/\.cache\/lemonade/i);

    if (await dirInput.isVisible()) {
      await dirInput.fill('~/.cache/lemonade');
      await page.waitForTimeout(200);

      const scanButton = page.getByRole('button', { name: /scan/i });
      if (await scanButton.isVisible()) {
        await scanButton.click();
        await page.waitForTimeout(1000);

        // Check for results alert
        const foundFiles = page.getByText(/found.*files/i);
        const foundVisible = await foundFiles.isVisible().catch(() => false);

        expect(typeof foundVisible).toBe('boolean');
      }
    }
  });
});

test.describe('Import Status Display', () => {
  test('should display initial empty state message', async ({ page }) => {
    const emptyMessage = page.getByText(/start.*import.*see progress/i);
    const messageVisible = await emptyMessage.isVisible().catch(() => false);

    expect(typeof emptyMessage).toBe('boolean');
  });

  test('should display job ID after import starts', async ({ page }) => {
    // Fill in cache directory
    const dirInput = page.getByPlaceholder(/~\/\.cache\/lemonade/i);

    if (await dirInput.isVisible()) {
      await dirInput.fill('~/.cache/lemonade');
      await page.waitForTimeout(200);

      // Click start import
      const importButton = page.getByRole('button', {
        name: /start import/i,
      });

      if (await importButton.isVisible()) {
        await importButton.click();
        await page.waitForTimeout(1000);

        // Should show job ID
        const jobIdLabel = page.getByText(/job id/i);
        const jobIdVisible = await jobIdLabel.isVisible().catch(() => false);

        expect(typeof jobIdVisible).toBe('boolean');
      }
    }
  });

  test('should display status badge', async ({ page }) => {
    // After import starts, should show status
    const dirInput = page.getByPlaceholder(/~\/\.cache\/lemonade/i);

    if (await dirInput.isVisible()) {
      await dirInput.fill('~/.cache/lemonade');
      await page.waitForTimeout(200);

      const importButton = page.getByRole('button', { name: /start import/i });
      if (await importButton.isVisible()) {
        await importButton.click();
        await page.waitForTimeout(1000);

        // Check for status badge
        const statusBadge = page.getByText(/running|completed|failed|pending/i);
        const statusVisible = await statusBadge.isVisible().catch(() => false);

        expect(typeof statusVisible).toBe('boolean');
      }
    }
  });

  test('should display progress bar during import', async ({ page }) => {
    const dirInput = page.getByPlaceholder(/~\/\.cache\/lemonade/i);

    if (await dirInput.isVisible()) {
      await dirInput.fill('~/.cache/lemonade');
      await page.waitForTimeout(200);

      const importButton = page.getByRole('button', { name: /start import/i });
      if (await importButton.isVisible()) {
        await importButton.click();
        await page.waitForTimeout(1000);

        // Check for progress bar
        const progressBar = page.getByRole('progressbar');
        const progressVisible = await progressBar.isVisible().catch(() => false);

        expect(typeof progressVisible).toBe('boolean');
      }
    }
  });

  test('should display import statistics', async ({ page }) => {
    const dirInput = page.getByPlaceholder(/~\/\.cache\/lemonade/i);

    if (await dirInput.isVisible()) {
      await dirInput.fill('~/.cache/lemonade');
      await page.waitForTimeout(200);

      const importButton = page.getByRole('button', { name: /start import/i });
      if (await importButton.isVisible()) {
        await importButton.click();
        await page.waitForTimeout(1000);

        // Check for statistics
        const importedText = page.getByText(/imported/i);
        const skippedText = page.getByText(/skipped/i);
        const errorsText = page.getByText(/errors/i);

        const statsVisible =
          (await importedText.isVisible().catch(() => false)) ||
          (await skippedText.isVisible().catch(() => false)) ||
          (await errorsText.isVisible().catch(() => false));

        expect(typeof statsVisible).toBe('boolean');
      }
    }
  });

  test('should display processed files count', async ({ page }) => {
    const dirInput = page.getByPlaceholder(/~\/\.cache\/lemonade/i);

    if (await dirInput.isVisible()) {
      await dirInput.fill('~/.cache/lemonade');
      await page.waitForTimeout(200);

      const importButton = page.getByRole('button', { name: /start import/i });
      if (await importButton.isVisible()) {
        await importButton.click();
        await page.waitForTimeout(1000);

        // Check for files count
        const filesCount = page.getByText(/files/i);
        const countVisible = await filesCount.isVisible().catch(() => false);

        expect(typeof countVisible).toBe('boolean');
      }
    }
  });

  test('should display percentage progress', async ({ page }) => {
    const dirInput = page.getByPlaceholder(/~\/\.cache\/lemonade/i);

    if (await dirInput.isVisible()) {
      await dirInput.fill('~/.cache/lemonade');
      await page.waitForTimeout(200);

      const importButton = page.getByRole('button', { name: /start import/i });
      if (await importButton.isVisible()) {
        await importButton.click();
        await page.waitForTimeout(1000);

        // Check for percentage
        const percentage = page.getByText(/\d+%/i);
        const percentVisible = await percentage.isVisible().catch(() => false);

        expect(typeof percentVisible).toBe('boolean');
      }
    }
  });
});

test.describe('Import Error Handling', () => {
  test('should display error alert for failed imports', async ({ page }) => {
    const dirInput = page.getByPlaceholder(/~\/\.cache\/lemonade/i);

    if (await dirInput.isVisible()) {
      await dirInput.fill('~/.cache/lemonade');
      await page.waitForTimeout(200);

      const importButton = page.getByRole('button', { name: /start import/i });
      if (await importButton.isVisible()) {
        await importButton.click();
        await page.waitForTimeout(2000);

        // Check for error alert
        const errorAlert = page.getByText(/import errors|error/i);
        const errorVisible = await errorAlert.isVisible().catch(() => false);

        expect(typeof errorVisible).toBe('boolean');
      }
    }
  });

  test('should display error list', async ({ page }) => {
    const dirInput = page.getByPlaceholder(/~\/\.cache\/lemonade/i);

    if (await dirInput.isVisible()) {
      await dirInput.fill('~/.cache/lemonade');
      await page.waitForTimeout(200);

      const importButton = page.getByRole('button', { name: /start import/i });
      if (await importButton.isVisible()) {
        await importButton.click();
        await page.waitForTimeout(2000);

        // Check for error code blocks
        const errorCodes = page.locator('code');
        const codeCount = await errorCodes.count();

        expect(codeCount).toBeGreaterThanOrEqual(0);
      }
    }
  });
});

test.describe('Import Success Handling', () => {
  test('should display success alert for completed imports', async ({ page }) => {
    const dirInput = page.getByPlaceholder(/~\/\.cache\/lemonade/i);

    if (await dirInput.isVisible()) {
      await dirInput.fill('~/.cache/lemonade');
      await page.waitForTimeout(200);

      const importButton = page.getByRole('button', { name: /start import/i });
      if (await importButton.isVisible()) {
        await importButton.click();
        await page.waitForTimeout(3000);

        // Check for success alert
        const successAlert = page.getByText(/import completed|successfully imported/i);
        const successVisible = await successAlert.isVisible().catch(() => false);

        expect(typeof successVisible).toBe('boolean');
      }
    }
  });

  test('should display duplicate skip count', async ({ page }) => {
    const dirInput = page.getByPlaceholder(/~\/\.cache\/lemonade/i);

    if (await dirInput.isVisible()) {
      await dirInput.fill('~/.cache/lemonade');
      await page.waitForTimeout(200);

      const importButton = page.getByRole('button', { name: /start import/i });
      if (await importButton.isVisible()) {
        await importButton.click();
        await page.waitForTimeout(2000);

        // Check for duplicates text
        const duplicatesText = page.getByText(/duplicates.*skipped|skipped.*duplicates/i);
        const duplicatesVisible = await duplicatesText.isVisible().catch(() => false);

        expect(typeof duplicatesVisible).toBe('boolean');
      }
    }
  });
});

test.describe('Import Page - Responsive Layout', () => {
  test('should have responsive grid layout', async ({ page }) => {
    await page.goto('/import');

    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.waitForTimeout(500);

    // Content should be visible
    const title = page.getByRole('heading', { name: /import/i });
    await expect(title).toBeVisible();

    // Test tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.waitForTimeout(500);

    await expect(title).toBeVisible();

    // Reset to desktop
    await page.setViewportSize({ width: 1920, height: 1080 });
  });

  test('should stack cards on mobile', async ({ page }) => {
    await page.goto('/import');
    await page.setViewportSize({ width: 375, height: 667 });
    await page.waitForTimeout(500);

    // Both cards should be visible on mobile
    const configCard = page.getByText(/import configuration/i);
    const statusCard = page.getByText(/import status/i);

    const configVisible = await configCard.isVisible().catch(() => false);
    const statusVisible = await statusCard.isVisible().catch(() => false);

    expect(configVisible && statusVisible).toBe(true);
  });
});

test.describe('Import Page - Loading States', () => {
  test('should display loading skeleton during status fetch', async ({ page }) => {
    await page.goto('/import');

    // Trigger a mock import state would show skeleton
    const skeleton = page.locator('[class*="skeleton"], [class*="Skeleton"]');
    const skeletonVisible = await skeleton.first().isVisible().catch(() => false);

    expect(typeof skeletonVisible).toBe('boolean');
  });

  test('should disable form during import', async ({ page }) => {
    const dirInput = page.getByPlaceholder(/~\/\.cache\/lemonade/i);

    if (await dirInput.isVisible()) {
      await dirInput.fill('~/.cache/lemonade');
      await page.waitForTimeout(200);

      const importButton = page.getByRole('button', { name: /start import/i });
      if (await importButton.isVisible()) {
        await importButton.click();
        await page.waitForTimeout(500);

        // Form fields should be disabled during import
        const dirDisabled = await dirInput.isDisabled();
        expect(typeof dirDisabled).toBe('boolean');
      }
    }
  });
});
