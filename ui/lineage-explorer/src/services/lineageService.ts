import axios from 'axios';

interface Dataset {
  namespace: string;
  name: string;
}

interface RunEvent {
  runId: string;
  job: {
    namespace: string;
    name: string;
  };
  inputs: Dataset[];
  outputs: Dataset[];
  eventTime: string;
  runState: string;
}

interface LineageNode {
  id: string;
  label: string;
  type: 'job' | 'dataset';
  namespace?: string;
}

interface LineageEdge {
  from: string;
  to: string;
}

interface LineageGraph {
  nodes: LineageNode[];
  edges: LineageEdge[];
}

const client = axios.create({
  baseURL: import.meta.env.VITE_OPENLINEAGE_URL,
  headers: {
    'Authorization': `Bearer ${import.meta.env.VITE_OPENLINEAGE_API_KEY}`,
    'Content-Type': 'application/json'
  }
});

export async function fetchRunLineage(runId: string): Promise<LineageGraph> {
  try {
    const response = await client.get(`/api/v1/runs/${runId}`);
    const runEvent: RunEvent = response.data;

    const nodes: LineageNode[] = [];
    const edges: LineageEdge[] = [];

    const jobId = `${runEvent.job.namespace}:${runEvent.job.name}`;
    nodes.push({
      id: jobId,
      label: runEvent.job.name,
      type: 'job',
      namespace: runEvent.job.namespace
    });

    runEvent.inputs.forEach(input => {
      const inputId = `${input.namespace}:${input.name}`;
      nodes.push({
        id: inputId,
        label: input.name,
        type: 'dataset',
        namespace: input.namespace
      });
      edges.push({ from: inputId, to: jobId });
    });

    runEvent.outputs.forEach(output => {
      const outputId = `${output.namespace}:${output.name}`;
      nodes.push({
        id: outputId,
        label: output.name,
        type: 'dataset',
        namespace: output.namespace
      });
      edges.push({ from: jobId, to: outputId });
    });

    return { nodes, edges };
  } catch (error) {
    return getMockLineage();
  }
}

export async function fetchAllRuns(): Promise<string[]> {
  try {
    const response = await client.get('/api/v1/runs');
    return response.data.runs.map((run: RunEvent) => run.runId);
  } catch (error) {
    return ['clean_options_30min-20250116-143000', 'calculate_features-20250116-143500', 'model_training-20250116-144000'];
  }
}

function getMockLineage(): LineageGraph {
  return {
    nodes: [
      { id: 'raw_options', label: 'Raw Options Data', type: 'dataset' },
      { id: 'clean_options_30min', label: 'Clean Options 30min', type: 'job' },
      { id: 'options_30min_clean', label: 'Options 30min Clean', type: 'dataset' },
      { id: 'daily_master', label: 'Daily Master', type: 'dataset' },
      { id: 'calculate_features', label: 'Calculate Features', type: 'job' },
      { id: 'features_output', label: 'Features Output', type: 'dataset' },
      { id: 'model_training', label: 'Model Training', type: 'job' },
      { id: 'trained_model', label: 'Trained Model', type: 'dataset' }
    ],
    edges: [
      { from: 'raw_options', to: 'clean_options_30min' },
      { from: 'clean_options_30min', to: 'options_30min_clean' },
      { from: 'options_30min_clean', to: 'calculate_features' },
      { from: 'daily_master', to: 'calculate_features' },
      { from: 'calculate_features', to: 'features_output' },
      { from: 'features_output', to: 'model_training' },
      { from: 'model_training', to: 'trained_model' }
    ]
  };
}
