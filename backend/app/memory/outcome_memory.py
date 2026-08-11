import sqlite3
import uuid
from typing import List, Optional
from app.schemas.profile import TaskType, Domain
from app.schemas.quality import QualityLabel
from app.schemas.memory import (
    ExecutionOutcomeRecord,
    HistoricalModelStats,
    AdaptiveRoutingSignal
)

class AdaptiveOutcomeMemory:
    """
    Adaptive Outcome Memory Service (Phase 10).
    
    Provides persistent SQLite storage for outcome events recorded across the GreenRoute pipeline.
    Calculates historical model performance statistics and exposes adaptive routing signals.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        # Keep an active connection if in-memory database is used so table persists across operations
        self._shared_conn = sqlite3.connect(db_path, check_same_thread=False) if db_path == ":memory:" else None
        if self._shared_conn:
            self._shared_conn.row_factory = sqlite3.Row
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._shared_conn:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create outcomes table schema if it does not exist."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS execution_outcomes (
                outcome_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                task_type TEXT NOT NULL,
                domain TEXT NOT NULL,
                model_id TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                routing_score REAL NOT NULL,
                routing_confidence TEXT NOT NULL,
                confidence_gap REAL NOT NULL,
                overall_quality_score REAL NOT NULL,
                quality_label TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                estimated_cost REAL NOT NULL,
                success INTEGER NOT NULL,
                is_fallback INTEGER NOT NULL
            )
        """)
        conn.commit()
        if not self._shared_conn:
            conn.close()

    def record_outcome(self, record: ExecutionOutcomeRecord) -> None:
        """Persist a single outcome record."""
        conn = self._get_connection()
        conn.execute("""
            INSERT INTO execution_outcomes (
                outcome_id, timestamp, task_type, domain, model_id, provider_name,
                routing_score, routing_confidence, confidence_gap, overall_quality_score,
                quality_label, latency_ms, estimated_cost, success, is_fallback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.outcome_id,
            record.timestamp,
            record.task_type.value if hasattr(record.task_type, "value") else str(record.task_type),
            record.domain.value if hasattr(record.domain, "value") else str(record.domain),
            record.model_id,
            record.provider_name,
            record.routing_score,
            record.routing_confidence,
            record.confidence_gap,
            record.overall_quality_score,
            record.quality_label.value if hasattr(record.quality_label, "value") else str(record.quality_label),
            record.latency_ms,
            record.estimated_cost,
            1 if record.success else 0,
            1 if record.is_fallback else 0
        ))
        conn.commit()
        if not self._shared_conn:
            conn.close()

    def get_recent_outcomes(self, limit: int = 50) -> List[ExecutionOutcomeRecord]:
        """Retrieve recent outcomes ordered by timestamp descending."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM execution_outcomes ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        records = [self._row_to_record(row) for row in rows]
        if not self._shared_conn:
            conn.close()
        return records

    def get_outcomes_for_model(self, model_id: str) -> List[ExecutionOutcomeRecord]:
        """Retrieve outcomes recorded for a specific model_id."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM execution_outcomes WHERE model_id = ? ORDER BY timestamp DESC",
            (model_id,)
        ).fetchall()
        records = [self._row_to_record(row) for row in rows]
        if not self._shared_conn:
            conn.close()
        return records

    def get_outcomes_for_task_and_domain(
        self,
        task_type: TaskType,
        domain: Domain
    ) -> List[ExecutionOutcomeRecord]:
        """Retrieve outcomes matching a specific task_type and domain combination."""
        task_str = task_type.value if hasattr(task_type, "value") else str(task_type)
        domain_str = domain.value if hasattr(domain, "value") else str(domain)

        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM execution_outcomes WHERE task_type = ? AND domain = ? ORDER BY timestamp DESC",
            (task_str, domain_str)
        ).fetchall()
        records = [self._row_to_record(row) for row in rows]
        if not self._shared_conn:
            conn.close()
        return records

    def calculate_model_stats(self, model_id: str) -> HistoricalModelStats:
        """
        Calculate historical aggregate performance statistics for a given model.
        Average quality score is computed over ALL recorded outcomes.
        """
        records = self.get_outcomes_for_model(model_id)

        if not records:
            return HistoricalModelStats(
                model_id=model_id,
                total_attempts=0,
                successful_attempts=0,
                success_rate=0.0,
                average_quality_score=0.0,
                average_latency_ms=0.0,
                average_cost=0.0,
                fallback_rate=0.0
            )

        total = len(records)
        succ = sum(1 for r in records if r.success)

        # Average quality score across ALL recorded outcomes
        avg_quality = sum(r.overall_quality_score for r in records) / total
        avg_lat = sum(r.latency_ms for r in records) / total
        avg_cost = sum(r.estimated_cost for r in records) / total
        fallback_cnt = sum(1 for r in records if r.is_fallback)

        return HistoricalModelStats(
            model_id=model_id,
            total_attempts=total,
            successful_attempts=succ,
            success_rate=round(succ / total, 4),
            average_quality_score=round(avg_quality, 4),
            average_latency_ms=round(avg_lat, 2),
            average_cost=round(avg_cost, 6),
            fallback_rate=round(fallback_cnt / total, 4)
        )

    def generate_adaptive_signal(
        self,
        model_id: str,
        task_type: Optional[TaskType] = None,
        domain: Optional[Domain] = None
    ) -> AdaptiveRoutingSignal:
        """
        Generate adaptive historical signal multiplier for future router integration.
        
        Empty History:
        If total_samples == 0:
            historical_quality_score = None
            historical_success_rate = None
            historical_fallback_rate = 0.0
            adaptive_weight_multiplier = 1.0 (neutral baseline)
        """
        records = self.get_outcomes_for_model(model_id)

        if task_type:
            task_str = task_type.value if hasattr(task_type, "value") else str(task_type)
            records = [r for r in records if r.task_type == task_str or r.task_type == task_type]
        if domain:
            dom_str = domain.value if hasattr(domain, "value") else str(domain)
            records = [r for r in records if r.domain == dom_str or r.domain == domain]

        if not records:
            return AdaptiveRoutingSignal(
                model_id=model_id,
                task_type=task_type,
                domain=domain,
                historical_quality_score=None,
                historical_success_rate=None,
                historical_fallback_rate=0.0,
                adaptive_weight_multiplier=1.0,
                total_samples=0
            )

        total = len(records)
        succ = sum(1 for r in records if r.success)

        # Average quality score across ALL recorded outcomes
        avg_qual = sum(r.overall_quality_score for r in records) / total
        succ_rate = succ / total
        fb_rate = sum(1 for r in records if r.is_fallback) / total

        # Compute adaptive preference multiplier
        mult = 1.0 + (avg_qual - 0.5) * 0.4 + (succ_rate - 0.5) * 0.4 - (fb_rate * 0.3)
        mult = max(0.5, min(1.5, mult))

        return AdaptiveRoutingSignal(
            model_id=model_id,
            task_type=task_type,
            domain=domain,
            historical_quality_score=round(avg_qual, 4),
            historical_success_rate=round(succ_rate, 4),
            historical_fallback_rate=round(fb_rate, 4),
            adaptive_weight_multiplier=round(mult, 4),
            total_samples=total
        )

    def _row_to_record(self, row: sqlite3.Row) -> ExecutionOutcomeRecord:
        return ExecutionOutcomeRecord(
            outcome_id=row["outcome_id"],
            timestamp=row["timestamp"],
            task_type=TaskType(row["task_type"]) if row["task_type"] in TaskType.__members__.values() else TaskType.UNKNOWN,
            domain=Domain(row["domain"]) if row["domain"] in Domain.__members__.values() else Domain.UNKNOWN,
            model_id=row["model_id"],
            provider_name=row["provider_name"],
            routing_score=row["routing_score"],
            routing_confidence=row["routing_confidence"],
            confidence_gap=row["confidence_gap"],
            overall_quality_score=row["overall_quality_score"],
            quality_label=QualityLabel(row["quality_label"]) if row["quality_label"] in QualityLabel.__members__.values() else QualityLabel.WEAK,
            latency_ms=row["latency_ms"],
            estimated_cost=row["estimated_cost"],
            success=bool(row["success"]),
            is_fallback=bool(row["is_fallback"])
        )
