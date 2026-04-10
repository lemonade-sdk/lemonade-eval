/**
 * ProtectedRoute — Redirects unauthenticated users to /login.
 * Includes a hydration guard to avoid false redirects while Zustand
 * persist middleware is rehydrating from localStorage.
 */

import { useEffect, useState } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { Center, Loader } from '@mantine/core';
import { useAuthStore } from '@/stores/authStore';

interface ProtectedRouteProps {
  children: React.ReactNode;
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const location = useLocation();
  const { user, token } = useAuthStore();
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    // Zustand persist rehydrates synchronously on the first render in most
    // environments, but we wait one microtask to ensure the store is settled
    // before making an auth decision.
    const unsub = useAuthStore.persist.onFinishHydration(() => {
      setHydrated(true);
    });

    // If already hydrated (e.g. synchronous rehydration), mark immediately
    if (useAuthStore.persist.hasHydrated()) {
      setHydrated(true);
    }

    return unsub;
  }, []);

  if (!hydrated) {
    return (
      <Center h="100vh">
        <Loader size="lg" />
      </Center>
    );
  }

  // User and token come from the persisted store; both must be present
  if (!user || !token) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <>{children}</>;
}

export default ProtectedRoute;
