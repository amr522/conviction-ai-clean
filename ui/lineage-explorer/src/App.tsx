import React, { useState, useEffect } from 'react';
import LineageGraph from './components/LineageGraph';
import { fetchAllRuns } from './services/lineageService';
import { pageViews } from './metrics';

export default function App() {
  const [runId, setRunId] = useState('');
  const [availableRuns, setAvailableRuns] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [availableTags] = useState<string[]>(['training', 'validation', 'production', 'backfill', 'drift']);

  useEffect(() => {
    pageViews.inc({ page: 'main' });
    fetchAllRuns().then(setAvailableRuns);
  }, []);

  const filteredRuns = availableRuns.filter(run => {
    const matchesSearch = run.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesTags = selectedTags.length === 0 || selectedTags.some(tag => run.includes(tag));
    return matchesSearch && matchesTags;
  });

  const toggleTag = (tag: string) => {
    setSelectedTags(prev =>
      prev.includes(tag)
        ? prev.filter(t => t !== tag)
        : [...prev, tag]
    );
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '20px' }}>
        Pipeline Lineage Explorer
      </h1>

      <div style={{ marginBottom: '20px' }}>
        {/* Search and Filter Controls */}
        <div style={{ marginBottom: '15px', display: 'flex', gap: '15px', alignItems: 'center', flexWrap: 'wrap' }}>
          <input
            type="text"
            placeholder="Search runs..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            style={{
              padding: '8px 12px',
              border: '1px solid #ddd',
              borderRadius: '4px',
              fontSize: '14px',
              minWidth: '200px'
            }}
          />

          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span style={{ fontSize: '14px', color: '#666' }}>Tags:</span>
            {availableTags.map(tag => (
              <button
                key={tag}
                onClick={() => toggleTag(tag)}
                style={{
                  padding: '4px 8px',
                  border: '1px solid #ddd',
                  borderRadius: '16px',
                  fontSize: '12px',
                  cursor: 'pointer',
                  backgroundColor: selectedTags.includes(tag) ? '#007bff' : '#f8f9fa',
                  color: selectedTags.includes(tag) ? 'white' : '#333'
                }}
              >
                {tag}
              </button>
            ))}
          </div>
        </div>

        <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>
          Select Run ID ({filteredRuns.length} runs):
        </label>
        <select
          value={runId}
          onChange={e => setRunId(e.target.value)}
          style={{
            padding: '8px 12px',
            border: '1px solid #ddd',
            borderRadius: '4px',
            fontSize: '14px',
            minWidth: '300px',
            marginRight: '10px'
          }}
        >
          <option value="">Select a run...</option>
          {filteredRuns.map(run => (
            <option key={run} value={run}>{run}</option>
          ))}
        </select>

        <span style={{ fontSize: '12px', color: '#666', marginLeft: '10px' }}>
          Or enter custom run ID:
        </span>
        <input
          type="text"
          placeholder="Enter run ID"
          value={runId}
          onChange={e => setRunId(e.target.value)}
          style={{
            padding: '8px 12px',
            border: '1px solid #ddd',
            borderRadius: '4px',
            fontSize: '14px',
            marginLeft: '10px',
            minWidth: '200px'
          }}
        />
      </div>

      {runId && (
        <div>
          <h2 style={{ fontSize: '18px', marginBottom: '10px' }}>
            Lineage for: {runId}
          </h2>
          <LineageGraph runId={runId} />
        </div>
      )}

      {!runId && (
        <div style={{
          padding: '40px',
          textAlign: 'center',
          color: '#666',
          border: '2px dashed #ddd',
          borderRadius: '8px'
        }}>
          Select a run ID to view its data lineage
        </div>
      )}
    </div>
  );
}
