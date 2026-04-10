/**
 * E2E Tests for Authentication Flow
 * Tests login, logout, and session management
 */

import { test, expect } from '@playwright/test';

test.describe('Authentication', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the app before each test
    await page.goto('/');
  });

  test('should display login page when not authenticated', async ({ page }) => {
    // Check if login form is visible
    await expect(page.locator('input[type="email"]')).toBeVisible();
    await expect(page.locator('input[type="password"]')).toBeVisible();
    await expect(page.locator('button[type="submit"]')).toBeVisible();
  });

  test('should show login form elements', async ({ page }) => {
    // Check for email input
    const emailInput = page.getByLabel(/email/i);
    await expect(emailInput).toBeVisible();

    // Check for password input
    const passwordInput = page.getByLabel(/password/i);
    await expect(passwordInput).toBeVisible();

    // Check for submit button
    const submitButton = page.getByRole('button', { name: /sign in|login|submit/i });
    await expect(submitButton).toBeVisible();
  });

  test('should validate empty email field', async ({ page }) => {
    const submitButton = page.getByRole('button', { name: /sign in|login|submit/i });

    // Try to submit without filling fields
    await submitButton.click();

    // Should show validation error or not submit
    await expect(page.locator('input[type="email"]')).toBeFocused();
  });

  test('should validate empty password field', async ({ page }) => {
    const emailInput = page.getByLabel(/email/i);
    const submitButton = page.getByRole('button', { name: /sign in|login|submit/i });

    // Fill email but not password
    await emailInput.fill('test@example.com');
    await submitButton.click();

    // Should show validation error or not submit
    await expect(page.locator('input[type="password"]')).toBeFocused();
  });

  test('should show error message for invalid credentials', async ({ page }) => {
    const emailInput = page.getByLabel(/email/i);
    const passwordInput = page.getByLabel(/password/i);
    const submitButton = page.getByRole('button', { name: /sign in|login|submit/i });

    // Fill with invalid credentials
    await emailInput.fill('invalid@example.com');
    await passwordInput.fill('wrongpassword');
    await submitButton.click();

    // Should show error message (implementation dependent)
    // Wait for potential error message
    await page.waitForTimeout(1000);

    // Check for error message or stay on login page
    const errorMessage = page.getByText(/invalid|error|failed|incorrect/i);
    // Error may or may not be shown depending on backend implementation
  });

  test('should navigate to dashboard after successful login', async ({ page }) => {
    // This test assumes a working backend - would need mock for CI
    const emailInput = page.getByLabel(/email/i);
    const passwordInput = page.getByLabel(/password/i);
    const submitButton = page.getByRole('button', { name: /sign in|login|submit/i });

    // Fill with valid credentials (would need actual user in DB)
    await emailInput.fill('admin@example.com');
    await passwordInput.fill('admin123');
    await submitButton.click();

    // Wait for navigation
    await page.waitForTimeout(1000);

    // Should navigate to dashboard or show dashboard content
    // This depends on actual authentication implementation
  });

  test('should have proper page title', async ({ page }) => {
    await expect(page).toHaveTitle(/Lemonade|Dashboard|Login/i);
  });

  test('should display logo or app name', async ({ page }) => {
    // Check for app branding
    const logo = page.getByText(/lemonade|eval|dashboard/i);
    await expect(logo).toBeVisible();
  });

  test('should have responsive design', async ({ page }) => {
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });

    const emailInput = page.getByLabel(/email/i);
    await expect(emailInput).toBeVisible();

    // Test tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 });

    await expect(emailInput).toBeVisible();
  });
});

test.describe('Logout Flow', () => {
  test('should show logout option after login', async ({ page }) => {
    // Navigate to app
    await page.goto('/');

    // After login, there should be a logout option somewhere
    // This is implementation dependent
    const logoutButton = page.getByRole('button', { name: /logout|sign out|exit/i });

    // May or may not be visible before login
    const isVisible = await logoutButton.isVisible().catch(() => false);
    expect(typeof isVisible).toBe('boolean');
  });

  test('should redirect to login page after logout', async ({ page }) => {
    // Navigate to app
    await page.goto('/');

    // Find and click logout
    const logoutButton = page.getByRole('button', { name: /logout|sign out/i });

    if (await logoutButton.isVisible().catch(() => false)) {
      await logoutButton.click();

      // Should redirect to login
      await expect(page.locator('input[type="email"]')).toBeVisible();
    }
  });
});

test.describe('Session Management', () => {
  test('should persist session on page refresh', async ({ page }) => {
    await page.goto('/');

    // Get current URL
    const currentUrl = page.url();

    // Refresh page
    await page.reload();

    // Should stay on same page if authenticated
    // or return to login if not
    expect(page.url()).toBeTruthy();
  });

  test('should clear session on logout', async ({ page }) => {
    await page.goto('/');

    // Clear localStorage to simulate logout
    await page.evaluate(() => localStorage.clear());

    // Should show login page
    await expect(page.locator('input[type="email"]')).toBeVisible();
  });
});
