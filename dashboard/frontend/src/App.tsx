/**
 * Main App Component with Routing
 */

import { Routes, Route, Navigate } from 'react-router-dom';
import { Box } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';

import AppShell from '@components/common/AppShell';
import DashboardPage from '@pages/dashboard/DashboardPage';
import ModelsPage from '@pages/models/ModelsPage';
import ModelDetailPage from '@pages/models/ModelDetailPage';
import RunsPage from '@pages/runs/RunsPage';
import RunDetailPage from '@pages/runs/RunDetailPage';
import ComparePage from '@pages/compare/ComparePage';
import ImportPage from '@pages/import/ImportPage';
import SettingsPage from '@pages/settings/SettingsPage';
import LoginPage from '@pages/auth/LoginPage';

function App() {
  const [mobileOpened, { toggle: toggleMobile }] = useDisclosure(false);
  const [desktopOpened, { toggle: toggleDesktop }] = useDisclosure(true);

  return (
    <AppShell mobileOpened={mobileOpened} desktopOpened={desktopOpened}>
      <Box component="main" id="main-content" p="md" style={{ flex: 1, overflow: 'auto' }}>
        <Routes>
          {/* Public routes */}
          <Route path="/login" element={<LoginPage />} />

          {/* Protected routes */}
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/models/:id" element={<ModelDetailPage />} />
          <Route path="/runs" element={<RunsPage />} />
          <Route path="/runs/:id" element={<RunDetailPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/import" element={<ImportPage />} />
          <Route path="/settings" element={<SettingsPage />} />

          {/* Catch all - redirect to dashboard */}
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </Box>
    </AppShell>
  );
}

export default App;
