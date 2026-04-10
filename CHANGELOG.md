# Changelog

All notable changes to the Lemonade Eval Dashboard project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Beta Release P1 Features

#### Backend Improvements

**WebSocket Exception Cleanup** (`backend/app/websocket.py`)
- Added proper try/finally blocks to ensure connection cleanup
- Implemented graceful handling of disconnect exceptions
- Added comprehensive logging for connection lifecycle events
- Improved error handling in `broadcast_to_run()` and `broadcast_evaluations()` methods
- Enhanced `handle_client_message()` with better error recovery

**API Documentation** (`dashboard/API.md`)
- Complete API reference with request/response examples
- OpenAPI-compatible endpoint documentation
- Usage examples for all API groups:
  - Health endpoints
  - Models API (CRUD operations)
  - Runs API (evaluation management)
  - Metrics API (performance and accuracy data)
  - Import API (YAML import operations)
  - WebSocket API (real-time updates)
- Authentication and rate limiting documentation
- Error codes reference

**Deployment Guide** (`dashboard/DEPLOYMENT.md`)
- Production deployment instructions
- Docker deployment with docker-compose
- Environment configuration guide
- Database migration procedures
- SSL/HTTPS setup with nginx and Let's Encrypt
- Monitoring and logging configuration
- Systemd service setup
- Security checklist

#### Frontend Improvements

**Polling Interval Optimization** (`frontend/src/hooks/`)
- Changed polling intervals from 5s/10s to 15s/30s for better performance
- Made intervals configurable via environment variables:
  - `VITE_POLLING_INTERVAL_FAST` (default: 30s)
  - `VITE_POLLING_INTERVAL_SLOW` (default: 15s)
  - `VITE_POLLING_INTERVAL_IMPORT` (default: 2s)
- Updated `.env.example` with polling configuration documentation

**Theme Management Unification** (`frontend/src/`)
- Removed dual theme management (Mantine + custom)
- Now using only Mantine's theme provider
- Theme persists across page refreshes via localStorage
- Proper synchronization between UI store and Mantine color scheme
- Fixed theme toggle functionality in AppShell and SettingsPage

**Accessibility Improvements** (WCAG 2.1 Level A)
- Added aria-labels to all icon buttons:
  - Theme toggle buttons
  - Navigation burgers
  - Action menu buttons
  - Copy and delete buttons
- Added skip-to-content link for keyboard navigation
- Fixed heading hierarchy (h1 → h2 → h3 → h4)
- Added proper role attributes where needed
- Improved keyboard navigation support
- Updated components:
  - `AppShell.tsx` - Skip link, aria-labels on burgers and action icons
  - `DataTable.tsx` - aria-labels on checkboxes and menu buttons
  - `SettingsPage.tsx` - aria-labels on theme toggle and API key actions
  - `App.tsx` - Added id="main-content" for skip link target

### Changed

- Reduced polling frequency to improve server performance
- Simplified theme management architecture
- Enhanced WebSocket connection reliability
- Improved error handling throughout the application

### Fixed

- WebSocket connection cleanup on disconnect
- Theme persistence across page refreshes
- Missing accessibility attributes on interactive elements

### Added - Dashboard UI/UX Session

- Benchmarks Page: new frontend page for visualizing benchmark sweep results across models
- Accuracy Page: new frontend page for accuracy metric comparison across models

### Fixed - Dashboard UI/UX Session

- Backend API server port standardized to 3001 across all configuration (was 8000)
- Frontend dev server URL updated to `localhost:5173` (Vite default; was `localhost:3000`)
- `AppShell.Main` wrapper added as required compound component for correct Mantine v7 layout rendering
- Auth login response now handled as direct OAuth2 token shape (not nested in wrapper)
- FastAPI route ordering corrected: static routes (`/benchmark/results`) registered before parameterized routes (`/{run_id}`)
- `MODEL_COLORS` undefined reference replaced with `SWEEP_PALETTE` constant
- Rules of Hooks violation resolved: hooks moved above early-return guards in `RunDetailPage`
- `RunStats` TypeScript interface corrected to match actual backend API response shape
- `IconBrandLemonade` (non-existent) replaced with `IconFlask` from `@tabler/icons-react`
- ES module re-export syntax fixed in `components/common/index.ts`
- FastAPI array `detail` format handled correctly in API client error processing
- `aiosqlite` and `pydantic[email]` added as required backend dependencies

## [1.0.0] - 2024-01-15

### Added

- Initial release
- FastAPI backend with PostgreSQL/SQLite support
- React frontend with Mantine UI components
- Real-time updates via WebSocket
- YAML import functionality
- Dashboard, Models, Runs, and Metrics pages
- Basic authentication

---

## Version History

- **Unreleased** - Beta release with P1 improvements
- **1.0.0** - Initial release


[Unreleased]: https://github.com/lemonade/lemonade-eval/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/lemonade/lemonade-eval/releases/tag/v1.0.0
