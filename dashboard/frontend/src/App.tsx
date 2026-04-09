/**
 * Main App Component with Routing
 */

import { Routes, Route, Navigate } from 'react-router-dom';
import { Box } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';

import AppShell from '@components/common/AppShell';
import { ProtectedRoute } from '@/components/common';
import DashboardPage from '@pages/dashboard/DashboardPage';
import ModelsPage from '@pages/models/ModelsPage';
import ModelDetailPage from '@pages/models/ModelDetailPage';
import RunsPage from '@pages/runs/RunsPage';
import RunDetailPage from '@pages/runs/RunDetailPage';
import ComparePage from '@pages/compare/ComparePage';
import ImportPage from '@pages/import/ImportPage';
import SettingsPage from '@pages/settings/SettingsPage';
import LoginPage from '@pages/auth/LoginPage';
import BenchmarksPage from '@pages/benchmarks/BenchmarksPage';
import AccuracyPage from '@pages/accuracy/AccuracyPage';

function App() {
  const [mobileOpened, { toggle: toggleMobile }] = useDisclosure(false);
  const [desktopOpened, { toggle: toggleDesktop }] = useDisclosure(true);

  return (
    <AppShell mobileOpened={mobileOpened} desktopOpened={desktopOpened} toggleMobile={toggleMobile} toggleDesktop={toggleDesktop}>
      <Box component="main" id="main-content">
        <Routes>
          {/* Public routes */}
          <Route path="/login" element={<LoginPage />} />

          {/* Protected routes */}
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<ProtectedRoute><DashboardPage /></ProtectedRoute>} />
          <Route path="/models" element={<ProtectedRoute><ModelsPage /></ProtectedRoute>} />
          <Route path="/models/:id" element={<ProtectedRoute><ModelDetailPage /></ProtectedRoute>} />
          <Route path="/runs" element={<ProtectedRoute><RunsPage /></ProtectedRoute>} />
          <Route path="/runs/:id" element={<ProtectedRoute><RunDetailPage /></ProtectedRoute>} />
          <Route path="/compare" element={<ProtectedRoute><ComparePage /></ProtectedRoute>} />
          <Route path="/benchmarks" element={<ProtectedRoute><BenchmarksPage /></ProtectedRoute>} />
          <Route path="/accuracy" element={<ProtectedRoute><AccuracyPage /></ProtectedRoute>} />
          <Route path="/import" element={<ProtectedRoute><ImportPage /></ProtectedRoute>} />
          <Route path="/settings" element={<ProtectedRoute><SettingsPage /></ProtectedRoute>} />

          {/* Catch all - redirect to dashboard */}
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </Box>
    </AppShell>
  );
}

export default App;
