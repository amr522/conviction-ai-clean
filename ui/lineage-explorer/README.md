# Pipeline Lineage Explorer

A lightweight React-based UI for exploring OpenLineage metadata and visualizing pipeline data flow as an interactive DAG.

## Setup

1. **Install dependencies:**
   ```bash
   cd ui/lineage-explorer
   npm install
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenLineage backend URL and API key
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```

4. **Open browser:**
   Navigate to `http://localhost:3000`

## Environment Variables

- `VITE_OPENLINEAGE_URL` - OpenLineage backend URL (e.g., `http://localhost:5000`)
- `VITE_OPENLINEAGE_API_KEY` - API key for authentication (optional)

## Features

- **Interactive DAG Visualization** - View pipeline data flow with vis-network
- **Run Selection** - Browse available runs or enter custom run IDs
- **Real-time Updates** - Fetches latest lineage metadata from OpenLineage
- **Mock Data Fallback** - Works offline with sample pipeline data

## Example Run IDs

When OpenLineage backend is not available, try these mock run IDs:
- `clean_options_30min-20250116-143000`
- `calculate_features-20250116-143500`
- `model_training-20250116-144000`

## Build for Production

```bash
npm run build
npm run preview
```

## Integration

The lineage explorer automatically fetches metadata from your OpenLineage backend and renders:
- **Jobs** (green boxes) - ETL steps, feature calculation, model training
- **Datasets** (blue ellipses) - Input/output data files and tables
- **Flow Direction** - Left-to-right data flow with arrows
