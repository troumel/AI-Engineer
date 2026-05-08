"""High-level orchestration service for NER extraction, training jobs, and model registry."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from app.models.schemas import (
    EntityPrediction,
    ExtractResponse,
    HealthCheckResponse,
    ModelVersionInfo,
    ModelsResponse,
    RolloutConfigResponse,
    TrainingJobRequest,
    TrainingJobResponse,
    TrainingJobStatusResponse,
)
from app.services.pattern_ner import PatternNERModel


@dataclass
class TrainingJobState:
    """Mutable training job state."""

    job_id: str
    requested_version: str
    status: str
    progress: int
    current_step: str
    created_version: str | None = None
    error: str | None = None


class NERService:
    """Coordinate model registry, extraction, training jobs, and A/B routing."""

    def __init__(
        self,
        models_directory: str,
        default_model_version: str,
        base_model_name: str,
        max_training_workers: int,
        default_candidate_percentage: float,
    ):
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.models_directory / "registry.json"
        self.default_model_version = default_model_version
        self.base_model_name = base_model_name
        self.default_candidate_percentage = default_candidate_percentage

        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_training_workers)
        self._jobs: dict[str, TrainingJobState] = {}
        self._models: dict[str, PatternNERModel] = {}
        self._registry = self._load_registry()
        self._load_or_seed_models()

    def extract_entities(
        self,
        text: str,
        model_version: str | None,
        use_ab_test: bool,
        audience_id: str | None,
    ) -> ExtractResponse:
        """Extract entities using explicit version selection or A/B routing."""
        selected_version, routing_strategy = self._select_version(
            text=text,
            explicit_version=model_version,
            use_ab_test=use_ab_test,
            audience_id=audience_id,
        )
        model = self._get_model(selected_version)
        entities = [
            EntityPrediction(
                text=entity.text,
                label=entity.label,
                start=entity.start,
                end=entity.end,
                score=entity.score,
            )
            for entity in model.extract(text)
        ]
        return ExtractResponse(
            text=text,
            model_version=selected_version,
            routing_strategy=routing_strategy,
            entities=entities,
        )

    def start_training_job(self, request: TrainingJobRequest) -> TrainingJobResponse:
        """Create and optionally schedule a training job."""
        if request.version_name in self._models:
            raise ValueError(f"Model version '{request.version_name}' already exists.")

        base_version = request.base_version or self.active_version
        if base_version not in self._models:
            raise ValueError(f"Base model version '{base_version}' does not exist.")

        job_id = uuid4().hex
        state = TrainingJobState(
            job_id=job_id,
            requested_version=request.version_name,
            status="queued",
            progress=0,
            current_step="Job created",
        )
        self._jobs[job_id] = state

        if request.run_async:
            self._executor.submit(self._run_training_job, job_id, request, base_version)
            return TrainingJobResponse(
                job_id=job_id,
                status=state.status,
                requested_version=request.version_name,
            )

        self._run_training_job(job_id, request, base_version)
        completed_state = self._jobs[job_id]
        return TrainingJobResponse(
            job_id=job_id,
            status=completed_state.status,
            requested_version=request.version_name,
        )

    def get_job_status(self, job_id: str) -> TrainingJobStatusResponse:
        """Return the status for a training job."""
        state = self._jobs.get(job_id)
        if state is None:
            raise ValueError(f"Training job '{job_id}' was not found.")

        return TrainingJobStatusResponse(
            job_id=state.job_id,
            requested_version=state.requested_version,
            status=state.status,
            progress=state.progress,
            current_step=state.current_step,
            created_version=state.created_version,
            error=state.error,
        )

    def list_models(self) -> ModelsResponse:
        """Return current model registry information."""
        models = [
            ModelVersionInfo(
                version_name=model.version_name,
                backend=model.backend_name,
                base_model_name=model.base_model_name,
                created_at=model.created_at,
                training_example_count=model.training_example_count,
                entity_labels=model.entity_labels,
                is_active=model.version_name == self.active_version,
            )
            for model in sorted(self._models.values(), key=lambda item: item.created_at)
        ]
        return ModelsResponse(
            active_version=self.active_version,
            rollout=self.get_rollout_config(),
            models=models,
        )

    def activate_model(self, version_name: str) -> None:
        """Set the active serving model version."""
        self._get_model(version_name)
        self._registry["active_version"] = version_name
        self._save_registry()

    def configure_rollout(
        self,
        primary_version: str,
        candidate_version: str,
        candidate_percentage: float,
    ) -> RolloutConfigResponse:
        """Configure A/B rollout between two model versions."""
        if primary_version == candidate_version:
            raise ValueError("Primary and candidate versions must be different.")
        self._get_model(primary_version)
        self._get_model(candidate_version)

        rollout = {
            "primary_version": primary_version,
            "candidate_version": candidate_version,
            "candidate_percentage": candidate_percentage,
        }
        self._registry["rollout"] = rollout
        self._save_registry()
        return self.get_rollout_config()

    def get_rollout_config(self) -> RolloutConfigResponse:
        """Return rollout configuration."""
        rollout = self._registry.get("rollout", {})
        return RolloutConfigResponse(
            primary_version=rollout.get("primary_version"),
            candidate_version=rollout.get("candidate_version"),
            candidate_percentage=float(rollout.get("candidate_percentage", 0.0)),
        )

    def get_health_status(self) -> HealthCheckResponse:
        """Return current health information."""
        running_jobs = sum(1 for job in self._jobs.values() if job.status == "running")
        return HealthCheckResponse(
            status="healthy",
            active_version=self.active_version,
            available_versions=len(self._models),
            running_jobs=running_jobs,
            rollout=self.get_rollout_config(),
        )

    @property
    def active_version(self) -> str:
        """Return the current active model version."""
        return str(self._registry["active_version"])

    def _run_training_job(
        self,
        job_id: str,
        request: TrainingJobRequest,
        base_version: str,
    ) -> None:
        state = self._jobs[job_id]
        try:
            self._update_job(state, status="running", progress=10, current_step="Validating dataset")

            examples_payload = [
                {
                    "text": example.text,
                    "entities": [entity.model_dump() for entity in example.entities],
                }
                for example in request.examples
            ]
            if not any(example["entities"] for example in examples_payload):
                raise ValueError("Training job requires at least one annotated entity.")

            self._update_job(state, progress=45, current_step="Building entity patterns")
            base_model = self._models[base_version]
            model = PatternNERModel.from_examples(
                version_name=request.version_name,
                base_model_name=base_model.base_model_name,
                examples=examples_payload,
                base_patterns=base_model.patterns,
            )

            self._update_job(state, progress=80, current_step="Persisting model version")
            model.save(self.models_directory)
            with self._lock:
                self._models[model.version_name] = model

            if request.auto_activate:
                self.activate_model(model.version_name)

            self._update_job(
                state,
                status="completed",
                progress=100,
                current_step="Training complete",
                created_version=model.version_name,
            )
        except Exception as exc:
            self._update_job(
                state,
                status="failed",
                progress=100,
                current_step="Training failed",
                error=str(exc),
            )

    def _update_job(
        self,
        state: TrainingJobState,
        status: str | None = None,
        progress: int | None = None,
        current_step: str | None = None,
        created_version: str | None = None,
        error: str | None = None,
    ) -> None:
        if status is not None:
            state.status = status
        if progress is not None:
            state.progress = progress
        if current_step is not None:
            state.current_step = current_step
        if created_version is not None:
            state.created_version = created_version
        if error is not None:
            state.error = error

    def _select_version(
        self,
        text: str,
        explicit_version: str | None,
        use_ab_test: bool,
        audience_id: str | None,
    ) -> tuple[str, str]:
        if explicit_version:
            self._get_model(explicit_version)
            return explicit_version, "explicit"

        rollout = self.get_rollout_config()
        if use_ab_test and rollout.primary_version and rollout.candidate_version:
            routing_key = audience_id or text
            ratio = self._routing_ratio(routing_key)
            if ratio < rollout.candidate_percentage:
                return rollout.candidate_version, "ab_test_candidate"
            return rollout.primary_version, "ab_test_primary"

        return self.active_version, "active"

    def _routing_ratio(self, value: str) -> float:
        digest = sha256(value.encode("utf-8")).hexdigest()[:8]
        return int(digest, 16) / 0xFFFFFFFF

    def _get_model(self, version_name: str) -> PatternNERModel:
        model = self._models.get(version_name)
        if model is None:
            raise ValueError(f"Model version '{version_name}' does not exist.")
        return model

    def _load_registry(self) -> dict[str, Any]:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        return {
            "active_version": self.default_model_version,
            "rollout": {
                "primary_version": self.default_model_version,
                "candidate_version": None,
                "candidate_percentage": self.default_candidate_percentage,
            },
        }

    def _save_registry(self) -> None:
        self.registry_path.write_text(json.dumps(self._registry, indent=2), encoding="utf-8")

    def _load_or_seed_models(self) -> None:
        for version_directory in self.models_directory.iterdir():
            if version_directory.is_dir() and (version_directory / "metadata.json").exists():
                model = PatternNERModel.load(version_directory)
                self._models[model.version_name] = model

        if not self._models:
            baseline = PatternNERModel.default_baseline(
                version_name=self.default_model_version,
                base_model_name=self.base_model_name,
            )
            baseline.save(self.models_directory)
            self._models[baseline.version_name] = baseline
            self._registry["active_version"] = baseline.version_name
            self._registry["rollout"] = {
                "primary_version": baseline.version_name,
                "candidate_version": None,
                "candidate_percentage": self.default_candidate_percentage,
            }
            self._save_registry()

        if self.active_version not in self._models:
            self._registry["active_version"] = next(iter(self._models))
            self._save_registry()