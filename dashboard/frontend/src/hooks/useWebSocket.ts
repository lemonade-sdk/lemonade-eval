/**
 * WebSocket Hook for Real-time Updates
 * Connects to backend WebSocket for live run status and metrics updates
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { useNotificationStore } from '@/stores/notificationStore';

const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL?.replace(/^http/, 'ws') || 'ws://localhost:8000';

export interface WSConfig {
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  onMessage?: (data: unknown) => void;
  onError?: (error: Event) => void;
  onOpen?: () => void;
  onClose?: () => void;
}

export interface UseWebSocketReturn {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  sendMessage: (data: unknown) => void;
  subscribe: (runId: string) => void;
  unsubscribe: () => void;
  connect: () => void;
  disconnect: () => void;
}

export function useWebSocket(runId?: string, config: WSConfig = {}): UseWebSocketReturn {
  const {
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
    onMessage,
    onError,
    onOpen,
    onClose,
  } = config;

  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const shouldReconnectRef = useRef(true);

  const addNotification = useNotificationStore((state) => state.addNotification);

  const cleanup = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;
    cleanup();

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
    setIsConnecting(false);
  }, [cleanup]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    cleanup();
    setIsConnecting(true);
    setError(null);

    try {
      const wsUrl = runId
        ? `${WS_BASE_URL}/ws/v1/evaluations?run_id=${runId}`
        : `${WS_BASE_URL}/ws/v1/evaluations`;

      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setIsConnected(true);
        setIsConnecting(false);
        reconnectAttemptsRef.current = 0;
        onOpen?.();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage?.(data);

          // Handle specific event types
          if (data.event_type === 'run_status') {
            addNotification({
              type: 'info',
              title: `Run ${data.status}`,
              message: data.message || `Run ${data.run_id} status updated to ${data.status}`,
              duration: 3000,
            });
          } else if (data.event_type === 'progress') {
            // Progress updates can be handled by parent component
          }
        } catch (parseError) {
          console.error('Failed to parse WebSocket message:', parseError);
        }
      };

      ws.onerror = (event) => {
        setError('WebSocket connection error');
        onError?.(event);
      };

      ws.onclose = () => {
        setIsConnected(false);
        setIsConnecting(false);
        onClose?.();

        // Attempt reconnection if not manually disconnected
        if (shouldReconnectRef.current && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      wsRef.current = ws;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect to WebSocket');
      setIsConnecting(false);
    }
  }, [runId, cleanup, onMessage, onError, onClose, reconnectInterval, maxReconnectAttempts, addNotification]);

  const sendMessage = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket not connected. Message not sent:', data);
    }
  }, []);

  const subscribe = useCallback((newRunId: string) => {
    sendMessage({ type: 'subscribe', run_id: newRunId });
  }, [sendMessage]);

  const unsubscribe = useCallback(() => {
    sendMessage({ type: 'unsubscribe' });
  }, [sendMessage]);

  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reconnect when runId changes
  useEffect(() => {
    if (runId !== undefined) {
      disconnect();
      connect();
    }
  }, [runId, connect, disconnect]);

  // Handle runId changes
  useEffect(() => {
    if (runId && isConnected) {
      subscribe(runId);
    }
  }, [runId, isConnected, subscribe]);

  return {
    isConnected,
    isConnecting,
    error,
    sendMessage,
    subscribe,
    unsubscribe,
    connect,
    disconnect,
  };
}

export default useWebSocket;
