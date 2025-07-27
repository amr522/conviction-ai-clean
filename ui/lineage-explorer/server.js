const express = require('express');
const path = require('path');
const { register } = require('prom-client');

const app = express();
const port = process.env.PORT || 80;

// Serve static files
app.use(express.static(path.join(__dirname, 'dist')));

// Metrics endpoint
app.get('/metrics', (req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(register.metrics());
});

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Serve React app
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(port, () => {
  console.log(`Lineage explorer running on port ${port}`);
});