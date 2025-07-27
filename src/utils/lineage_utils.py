"""OpenLineage utilities for data lineage tracking."""

import os
from datetime import datetime
from typing import List, Optional

from openlineage.client import OpenLineageClient
from openlineage.client.dataset import Dataset
from openlineage.client.facet import SourceCodeLocationJobFacet, SqlJobFacet
from openlineage.client.run import Job, Run, RunEvent, RunState


class LineageTracker:
    """Tracks data lineage for ETL operations."""

    def __init__(self, namespace: str = "conviction_ai"):
        self.client = (
            OpenLineageClient.from_environment()
            if os.getenv("OPENLINEAGE_URL")
            else None
        )
        self.namespace = namespace
        self.current_run = None

    def start_run(
        self,
        job_name: str,
        inputs: List[str],
        outputs: List[str],
        run_id: Optional[str] = None,
        sql: Optional[str] = None,
    ) -> Optional[Run]:
        """Start a lineage run."""
        if not self.client:
            return None

        run_id = run_id or f"{job_name}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

        job_facets = {}
        if sql:
            job_facets["sql"] = SqlJobFacet(query=sql)

        event = RunEvent(
            runId=run_id,
            job=Job(namespace=self.namespace, name=job_name, facets=job_facets),
            inputs=[Dataset(namespace="s3", name=inp) for inp in inputs],
            outputs=[Dataset(namespace="s3", name=out) for out in outputs],
            eventTime=datetime.utcnow().isoformat(),
            runState=RunState.START,
        )

        self.current_run = self.client.emit(event)
        return self.current_run

    def complete_run(self, success: bool = True):
        """Complete the current lineage run."""
        if not self.client or not self.current_run:
            return

        state = RunState.COMPLETE if success else RunState.FAIL
        event = self.current_run.event._replace(
            eventTime=datetime.utcnow().isoformat(), runState=state
        )

        self.client.emit(event)
        self.current_run = None


def track_lineage(
    job_name: str, inputs: List[str], outputs: List[str], sql: Optional[str] = None
):
    """Decorator for tracking function lineage."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = LineageTracker()
            tracker.start_run(job_name, inputs, outputs, sql=sql)
            try:
                result = func(*args, **kwargs)
                tracker.complete_run(success=True)
                return result
            except Exception as e:
                tracker.complete_run(success=False)
                raise e

        return wrapper

    return decorator
