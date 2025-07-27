import React, { useEffect, useRef } from 'react';
import { Network } from 'vis-network';
import { fetchRunLineage } from '../services/lineageService';

interface Props { 
  runId: string; 
}

export default function LineageGraph({ runId }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!runId || !containerRef.current) return;

    fetchRunLineage(runId).then(graph => {
      const nodes = graph.nodes.map(n => ({
        id: n.id,
        label: n.label,
        color: n.type === 'job' ? '#4CAF50' : '#2196F3',
        shape: n.type === 'job' ? 'box' : 'ellipse'
      }));
      
      const edges = graph.edges.map(e => ({
        from: e.from,
        to: e.to,
        arrows: 'to'
      }));

      new Network(
        containerRef.current!,
        { nodes, edges },
        {
          layout: { 
            hierarchical: { 
              direction: 'LR',
              sortMethod: 'directed'
            }
          },
          physics: false,
          nodes: {
            font: { size: 14 },
            margin: 10
          },
          edges: {
            color: '#666'
          }
        }
      );
    });
  }, [runId]);

  return (
    <div 
      ref={containerRef} 
      style={{ 
        width: '100%', 
        height: '600px',
        border: '1px solid #ddd',
        borderRadius: '4px'
      }} 
    />
  );
}