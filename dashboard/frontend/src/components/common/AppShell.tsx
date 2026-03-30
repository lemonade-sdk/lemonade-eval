/**
 * Main App Shell with Sidebar Navigation
 */

import { ReactNode } from 'react';
import {
  AppShell,
  Burger,
  Group,
  Skeleton,
  Center,
  Stack,
  UnstyledButton,
  Text,
  rem,
  useMantineColorScheme,
  ActionIcon,
  Tooltip,
} from '@mantine/core';
import { useDisclosure, useHover } from '@mantine/hooks';
import {
  IconDashboard,
  IconCpu,
  IconListDetails,
  IconChartBar,
  IconUpload,
  IconSettings,
  IconLogout,
  IconSun,
  IconMoon,
  IconBrandLemonade,
} from '@tabler/icons-react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useUIStore } from '@/stores/uiStore';
import { useAuthStore } from '@/stores/authStore';

interface AppShellProps {
  children: ReactNode;
  mobileOpened: boolean;
  desktopOpened: boolean;
}

const navLinks = [
  { path: '/dashboard', label: 'Dashboard', icon: IconDashboard },
  { path: '/models', label: 'Models', icon: IconCpu },
  { path: '/runs', label: 'Runs', icon: IconListDetails },
  { path: '/compare', label: 'Compare', icon: IconChartBar },
  { path: '/import', label: 'Import', icon: IconUpload },
  { path: '/settings', label: 'Settings', icon: IconSettings },
] as const;

export default function AppShellComponent({ children, mobileOpened, desktopOpened }: AppShellProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const { setColorScheme, colorScheme } = useMantineColorScheme();
  const { user, logout } = useAuthStore();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const handleThemeToggle = () => {
    const newScheme = colorScheme === 'light' ? 'dark' : 'light';
    setColorScheme(newScheme);
    // Also update the UI store for persistence
    useUIStore.getState().setColorScheme(newScheme);
  };

  return (
    <React.Fragment>
      {/* Skip to content link for accessibility */}
      <a
        href="#main-content"
        className="skip-link"
        style={{
          position: 'absolute',
          left: '-10000px',
          top: 'auto',
          width: '1px',
          height: '1px',
          overflow: 'hidden',
          zIndex: 999,
        }}
        onFocus={(e) => {
          e.currentTarget.style.position = 'fixed';
          e.currentTarget.style.left = '50%';
          e.currentTarget.style.top = '0';
          e.currentTarget.style.width = 'auto';
          e.currentTarget.style.height = 'auto';
          e.currentTarget.style.transform = 'translateX(-50%)';
          e.currentTarget.style.padding = '1rem';
          e.currentTarget.style.backgroundColor = 'var(--mantine-color-blue-6)';
          e.currentTarget.style.color = 'white';
          e.currentTarget.style.borderRadius = '0 0 8px 8px';
        }}
        onBlur={(e) => {
          e.currentTarget.style.position = 'absolute';
          e.currentTarget.style.left = '-10000px';
          e.currentTarget.style.top = 'auto';
          e.currentTarget.style.width = '1px';
          e.currentTarget.style.height = '1px';
          e.currentTarget.style.transform = 'none';
          e.currentTarget.style.padding = '0';
          e.currentTarget.style.backgroundColor = 'transparent';
          e.currentTarget.style.color = 'inherit';
          e.currentTarget.style.borderRadius = '0';
        }}
      >
        Skip to main content
      </a>

      <AppShell
        header={{ height: 60 }}
        navbar={{
          width: 260,
          breakpoint: 'sm',
          collapsed: { mobile: !mobileOpened, desktop: !desktopOpened },
        }}
        padding="md"
      >
        <AppShell.Header withBorder={false} bg="var(--mantine-color-body)">
          <Group h="100%" px="md" justify="space-between">
            <Group>
              <Burger
                opened={mobileOpened}
                onClick={() => {}}
                hiddenFrom="sm"
                size="sm"
                aria-label="Toggle mobile navigation"
              />
              <Burger
                opened={desktopOpened}
                onClick={() => {}}
                visibleFrom="sm"
                size="sm"
                aria-label="Toggle sidebar"
              />
              <Group gap="xs">
                <IconBrandLemonade size={28} color="var(--mantine-color-blue-6)" />
                <Text fw={700} size="lg" visibleFrom="sm">
                  Lemonade Eval
                </Text>
              </Group>
            </Group>
            <Group gap="xs">
              <Tooltip label={`Switch to ${colorScheme === 'light' ? 'dark' : 'light'} mode`}>
                <ActionIcon
                  variant="subtle"
                  onClick={handleThemeToggle}
                  aria-label={`Switch to ${colorScheme === 'light' ? 'dark' : 'light'} mode`}
                >
                  {colorScheme === 'light' ? <IconMoon size={18} /> : <IconSun size={18} />}
                </ActionIcon>
              </Tooltip>
              {user && (
                <Group gap="xs" visibleFrom="sm">
                  <Text size="sm" c="dimmed">
                    {user.name}
                  </Text>
                  <ActionIcon variant="subtle" onClick={handleLogout} aria-label="Logout">
                    <IconLogout size={18} />
                  </ActionIcon>
                </Group>
              )}
            </Group>
          </Group>
        </AppShell.Header>

        <AppShell.Navbar p="md">
          <Stack gap="xs">
            {navLinks.map((link) => {
              const isActive = location.pathname === link.path ||
                (link.path !== '/dashboard' && location.pathname.startsWith(link.path));
              return <NavLink
                key={link.path}
                icon={link.icon}
                label={link.label}
                to={link.path}
                active={isActive}
              />;
            })}
          </Stack>

          <Stack gap="xs" style={{ marginTop: 'auto' }}>
            <NavLink
              icon={IconLogout}
              label="Logout"
              onClick={handleLogout}
              active={false}
            />
          </Stack>
        </AppShell.Navbar>

        {children}
      </AppShell>
    </React.Fragment>
  );
}

interface NavLinkProps {
  icon: React.ComponentType<{ size: number }>;
  label: string;
  to?: string;
  active?: boolean;
  onClick?: () => void;
}

function NavLink({ icon: Icon, label, to, active, onClick }: NavLinkProps) {
  const { hovered, ref } = useHover();

  const content = (
    <UnstyledButton
      ref={ref}
      component={to ? Link : 'button'}
      to={to}
      onClick={onClick}
      w="100%"
      p="sm"
      style={{
        borderRadius: 'var(--mantine-radius-md)',
        backgroundColor: active
          ? 'var(--mantine-color-blue-light)'
          : hovered
            ? 'var(--mantine-color-default-hover)'
            : undefined,
        color: active ? 'var(--mantine-color-blue-6)' : undefined,
        transition: 'background-color 150ms ease',
      }}
    >
      <Group gap="sm">
        <Icon size={20} stroke={1.5} />
        <Text size="sm" fw={active ? 600 : 400}>
          {label}
        </Text>
      </Group>
    </UnstyledButton>
  );

  return content;
}
