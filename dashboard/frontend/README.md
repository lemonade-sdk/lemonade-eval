# Lemonade Eval Dashboard - Frontend

A modern, type-safe React TypeScript dashboard for visualizing and comparing LLM/VLM evaluation results.

## Tech Stack

- **Framework:** React 18 + TypeScript
- **Build Tool:** Vite
- **UI Library:** Mantine v7
- **State Management:** Zustand
- **Data Fetching:** TanStack Query (React Query)
- **Tables:** TanStack Table
- **Charts:** Recharts
- **Routing:** React Router v6
- **Forms:** React Hook Form
- **Validation:** Zod
- **Testing:** Vitest + React Testing Library

## Features

- Dashboard overview with key metrics and recent runs
- Models management with search and filtering
- Runs listing with pagination and filters
- Side-by-side model/run comparison
- YAML import with progress tracking
- Real-time updates via WebSocket
- Dark mode support
- Responsive design
- Full TypeScript type safety

## Getting Started

### Prerequisites

- Node.js 18+
- npm or pnpm
- Backend API running on port 8000

### Installation

1. Install dependencies:
```bash
npm install
```

2. Copy environment file:
```bash
cp .env.example .env
```

3. Update `.env` with your backend URL if different from default.

### Development

Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`.

### Building for Production

```bash
npm run build
npm run preview
```

### Testing

Run unit tests:
```bash
npm run test
```

Run tests with UI:
```bash
npm run test:ui
```

Run tests with coverage:
```bash
npm run test:coverage
```

## Project Structure

```
src/
├── api/           # API client and methods
├── assets/        # Static assets
├── components/    # Reusable components
│   ├── charts/    # Chart components (Line, Bar, Radar)
│   ├── common/    # Common UI components
│   ├── forms/     # Form components
│   └── metrics/   # Metric display components
├── hooks/         # Custom React hooks
├── pages/         # Page components
│   ├── auth/      # Authentication pages
│   ├── compare/   # Comparison page
│   ├── dashboard/ # Dashboard page
│   ├── import/    # Import page
│   ├── models/    # Models pages
│   ├── runs/      # Runs pages
│   └── settings/  # Settings page
├── stores/        # Zustand stores
├── tests/         # Test files
├── types/         # TypeScript types
├── utils/         # Utility functions
├── App.tsx        # Main app component
└── main.tsx       # Entry point
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |
| `npm run format` | Format code with Prettier |
| `npm run test` | Run tests |
| `npm run test:ui` | Run tests with UI |
| `npm run type-check` | Type check without emitting |

## API Integration

The frontend connects to the backend API at `http://localhost:8000` by default.

### API Endpoints

- `GET /api/v1/models` - List models
- `GET /api/v1/runs` - List runs
- `GET /api/v1/metrics` - List metrics
- `POST /api/v1/import/yaml` - Import YAML files
- `WS /ws/v1/evaluations` - WebSocket for real-time updates

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | Backend API URL | `http://localhost:8000` |
| `VITE_WS_BASE_URL` | WebSocket URL | `ws://localhost:8000` |

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

MIT
