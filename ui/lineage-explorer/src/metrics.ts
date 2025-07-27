import { register, Counter, Histogram, collectDefaultMetrics } from 'prom-client';

// Collect default metrics
collectDefaultMetrics({ register });

// Custom metrics
export const pageViews = new Counter({
  name: 'lineage_page_views_total',
  help: 'Total number of page views',
  labelNames: ['page']
});

export const runQueries = new Counter({
  name: 'lineage_run_queries_total',
  help: 'Total number of run queries',
  labelNames: ['status']
});

export const graphRenderTime = new Histogram({
  name: 'lineage_graph_render_duration_seconds',
  help: 'Time taken to render lineage graph',
  buckets: [0.1, 0.5, 1, 2, 5]
});

// Register metrics
register.registerMetric(pageViews);
register.registerMetric(runQueries);
register.registerMetric(graphRenderTime);

export { register };