/**
 * Error handling utilities
 * Extract error codes from API responses and map to user-friendly messages
 */

import { ApiErrorClass } from '@/types';
import type { ApiError } from '@/types';

/**
 * Error code to user-friendly message mapping
 */
const ERROR_CODE_MESSAGES: Record<string, string> = {
  // Authentication errors
  'invalid_credentials': 'The email or password you entered is incorrect.',
  'token_expired': 'Your session has expired. Please log in again.',
  'invalid_token': 'Invalid authentication token. Please log in again.',
  'unauthorized': 'You are not authorized to perform this action.',
  'forbidden': 'Access denied. This action requires elevated privileges.',
  'account_inactive': 'This account has been deactivated.',
  'email_not_found': 'No account found with this email address.',
  'email_already_exists': 'An account with this email already exists.',
  'password_too_weak': 'Password does not meet security requirements.',

  // Validation errors
  'validation_error': 'Please check your input and try again.',
  'invalid_input': 'The provided input is invalid.',
  'required_field': 'This field is required.',
  'invalid_format': 'The format of this field is incorrect.',

  // Resource errors
  'not_found': 'The requested resource was not found.',
  'already_exists': 'This resource already exists.',
  'conflict': 'There is a conflict with the current operation.',
  'locked': 'This resource is currently locked.',

  // Server errors
  'internal_error': 'An unexpected error occurred. Please try again later.',
  'service_unavailable': 'The service is temporarily unavailable. Please try again later.',
  'timeout': 'The request took too long. Please try again.',
  'network_error': 'Network error. Please check your connection.',

  // Rate limiting
  'rate_limit_exceeded': 'Too many requests. Please wait a moment and try again.',
  'quota_exceeded': 'You have exceeded your usage quota.',
};

/**
 * Default error messages by status code
 */
const STATUS_CODE_MESSAGES: Record<number, string> = {
  400: 'Bad request. Please check your input and try again.',
  401: 'Authentication required. Please log in.',
  403: 'You do not have permission to perform this action.',
  404: 'The requested resource was not found.',
  409: 'Conflict. The resource already exists.',
  422: 'Validation error. Please check your input.',
  429: 'Too many requests. Please try again later.',
  500: 'An unexpected error occurred. Please try again later.',
  502: 'Bad gateway. The service is temporarily unavailable.',
  503: 'Service unavailable. Please try again later.',
  504: 'Gateway timeout. Please try again later.',
};

/**
 * Generate a short error code from status and message
 */
export function generateErrorCode(status: number, message?: string): string {
  const statusPrefix = `ERR${status}`;

  if (message) {
    // Extract key words from message to create code
    const words = message.toLowerCase().replace(/[^a-z\s]/g, '').split(/\s+/);
    const significantWords = words.filter(
      (w) => w.length > 3 && !['the', 'and', 'for', 'with', 'this', 'that', 'have', 'been', 'are', 'was', 'were'].includes(w)
    );

    if (significantWords.length > 0) {
      const codeSuffix = significantWords.slice(0, 2).map(w => w.substring(0, 4)).join('_');
      return `${statusPrefix}_${codeSuffix}`.toUpperCase();
    }
  }

  return `${statusPrefix}_UNKNOWN`;
}

/**
 * Get user-friendly message for an error code
 */
export function getMessageForErrorCode(errorCode: string): string {
  const normalizedCode = errorCode.toLowerCase().replace(/[^a-z0-9_]/g, '_');

  // Direct match
  if (ERROR_CODE_MESSAGES[normalizedCode]) {
    return ERROR_CODE_MESSAGES[normalizedCode];
  }

  // Partial match
  for (const [code, message] of Object.entries(ERROR_CODE_MESSAGES)) {
    if (normalizedCode.includes(code) || code.includes(normalizedCode)) {
      return message;
    }
  }

  return null;
}

/**
 * Extract error information from API error
 */
export interface ErrorInfo {
  code: string;
  message: string;
  userMessage: string;
  status: number;
  details?: Record<string, unknown>;
  copyText: string;
}

/**
 * Parse an API error into structured error info
 */
export function parseApiError(error: unknown): ErrorInfo {
  // Default error
  const defaultError: ErrorInfo = {
    code: 'ERR_UNKNOWN',
    message: 'An unexpected error occurred',
    userMessage: 'An unexpected error occurred. Please try again.',
    status: 500,
    copyText: 'ERR_UNKNOWN: An unexpected error occurred',
  };

  if (!(error instanceof Error)) {
    return defaultError;
  }

  // Check if it's our ApiErrorClass
  if (error instanceof ApiErrorClass) {
    const errorCode = error.code || generateErrorCode(error.status, error.message);
    const userMessage = getMessageForErrorCode(errorCode) || error.message;

    return {
      code: errorCode,
      message: error.message,
      userMessage,
      status: error.status,
      details: error.details,
      copyText: `${errorCode}: ${error.message}`,
    };
  }

  // Handle generic errors
  return {
    code: 'ERR_EXCEPTION',
    message: error.message,
    userMessage: error.message || 'An unexpected error occurred',
    status: 500,
    copyText: `ERR_EXCEPTION: ${error.message}`,
  };
}

/**
 * Format error for display in notifications
 */
export interface NotificationError {
  title: string;
  message: string;
  errorCode?: string;
}

/**
 * Format error for notification display
 */
export function formatErrorForNotification(error: ErrorInfo): NotificationError {
  return {
    title: 'Error',
    message: `${error.userMessage}`,
    errorCode: error.code,
  };
}

/**
 * Copy error code to clipboard
 */
export async function copyErrorCodeToClipboard(errorCode: string, message: string): Promise<boolean> {
  const copyText = `${errorCode}: ${message}`;

  try {
    await navigator.clipboard.writeText(copyText);
    return true;
  } catch {
    // Fallback for older browsers
    const textArea = document.createElement('textarea');
    textArea.value = copyText;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    document.body.appendChild(textArea);
    textArea.select();

    try {
      document.execCommand('copy');
      document.body.removeChild(textArea);
      return true;
    } catch {
      document.body.removeChild(textArea);
      return false;
    }
  }
}

/**
 * Get field-specific errors from API error details
 */
export function getFieldErrors(error: unknown): Record<string, string> {
  const fieldErrors: Record<string, string> = {};

  if (error instanceof ApiErrorClass && error.details) {
    // Handle FastAPI validation errors
    const details = error.details as Record<string, unknown>;

    if (Array.isArray(details.detail)) {
      for (const detail of details.detail) {
        if (detail.loc && Array.isArray(detail.loc) && detail.msg) {
          const field = detail.loc[detail.loc.length - 1] as string;
          fieldErrors[field] = detail.msg as string;
        }
      }
    }

    // Handle field errors in errors array
    if (Array.isArray(details.errors)) {
      for (const err of details.errors) {
        if (err.field && err.message) {
          fieldErrors[err.field as string] = err.message as string;
        }
      }
    }
  }

  return fieldErrors;
}

/**
 * Error handler for API calls
 * Can be used with try-catch blocks
 */
export class ErrorHandler {
  private onError?: (errorInfo: ErrorInfo) => void;

  constructor(onError?: (errorInfo: ErrorInfo) => void) {
    this.onError = onError;
  }

  async handle<T>(promise: Promise<T>): Promise<T | null> {
    try {
      return await promise;
    } catch (error) {
      const errorInfo = parseApiError(error);
      this.onError?.(errorInfo);
      return null;
    }
  }

  handleSync<T>(fn: () => T): T | null {
    try {
      return fn();
    } catch (error) {
      const errorInfo = parseApiError(error);
      this.onError?.(errorInfo);
      return null;
    }
  }
}

export default {
  parseApiError,
  formatErrorForNotification,
  copyErrorCodeToClipboard,
  getFieldErrors,
  generateErrorCode,
  getMessageForErrorCode,
  ErrorHandler,
};
