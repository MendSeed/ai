"""
MendSeed AI — Transactional Outbox State Machine Pattern.

Extracts the outbox asynchronous processing model from research notebooks.
Demonstrates decoupled database transactions:
Citizen grievance ingestion is instant and atomic.
LLM inference / moderation executes strictly outside database write locks.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, List, Literal, Optional

from .contracts import GrievanceAnalysis

logger = logging.getLogger("mendseed_ai.outbox")

ReportState = Literal[
    "RECEIVED",
    "MODERATION_PENDING",
    "APPROVED",
    "REJECTED_SPAM",
    "FAILED_VALIDATION",
    "PUBLISHED"
]


@dataclass
class OutboxTask:
    """A persistent unit of asynchronous work scheduled inside a DB transaction."""
    id: str
    report_id: str
    action: str = "PROCESS_AND_CATEGORIZE"
    status: Literal["PENDING", "IN_PROGRESS", "PROCESSED", "FAILED"] = "PENDING"
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None


@dataclass
class ReportRecord:
    """Simulated authoritative database row for a citizen report."""
    id: str
    text: str
    state: ReportState = "RECEIVED"
    metadata: Dict[str, Any] = field(default_factory=dict)
    structured_analysis: Optional[Dict[str, Any]] = None
    history: List[str] = field(default_factory=list)


class TransactionalOutboxManager:
    """
    In-memory reference implementation of MendSeed's Outbox pattern.
    Ensures zero blocking ML calls inside write transactions.
    """

    def __init__(self):
        self.reports_table: Dict[str, ReportRecord] = {}
        self.outbox_queue: List[OutboxTask] = []

    def ingest_report_transaction(
        self,
        report_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ReportRecord:
        """
        Step 1: Ingest report atomically and enqueue outbox task.
        NO network calls or LLM inference run here.
        """
        record = ReportRecord(
            id=report_id,
            text=text,
            state="RECEIVED",
            metadata=metadata or {},
            history=["RECEIVED"]
        )
        self.reports_table[report_id] = record

        task = OutboxTask(
            id=f"task_{report_id}",
            report_id=report_id,
            action="PROCESS_AND_CATEGORIZE",
            status="PENDING"
        )
        self.outbox_queue.append(task)
        return record

    async def execute_outbox_worker(
        self,
        inference_func: Callable[[str], Any]
    ) -> int:
        """
        Step 2: Background worker runs outside DB write lock context.
        Calls inference service asynchronously and updates records.
        """
        processed_count = 0
        for task in self.outbox_queue:
            if task.status != "PENDING":
                continue

            record = self.reports_table.get(task.report_id)
            if not record:
                task.status = "FAILED"
                task.error_message = f"Report {task.report_id} not found"
                continue

            task.status = "IN_PROGRESS"
            record.state = "MODERATION_PENDING"
            record.history.append("MODERATION_PENDING")

            try:
                # Call inference function
                import inspect
                if inspect.iscoroutinefunction(inference_func):
                    result = await inference_func(record.text)
                else:
                    result = inference_func(record.text)

                if result:
                    record.structured_analysis = (
                        result.model_dump() if hasattr(result, "model_dump") else result
                    )
                    record.state = "APPROVED"
                    record.history.append("APPROVED")
                    task.status = "PROCESSED"
                    processed_count += 1
                else:
                    record.state = "FAILED_VALIDATION"
                    record.history.append("FAILED_VALIDATION")
                    task.status = "FAILED"
            except Exception as e:
                logger.error(f"Worker task error for {task.id}: {e}")
                task.retry_count += 1
                if task.retry_count >= task.max_retries:
                    record.state = "FAILED_VALIDATION"
                    record.history.append("FAILED_VALIDATION")
                    task.status = "FAILED"
                    task.error_message = str(e)
                else:
                    task.status = "PENDING"

        return processed_count
