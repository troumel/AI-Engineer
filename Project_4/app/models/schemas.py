"""Pydantic request and response models for the NER API."""

from pydantic import BaseModel, Field, field_validator


class EntitySpan(BaseModel):
    """Annotated entity span for training data."""

    start: int = Field(..., ge=0, description="Start character offset.")
    end: int = Field(..., gt=0, description="End character offset.")
    label: str = Field(..., min_length=1, max_length=64, description="Entity label.")


class TrainingExample(BaseModel):
    """One annotated training sample."""

    text: str = Field(..., min_length=1, description="Training text.")
    entities: list[EntitySpan] = Field(default_factory=list, description="Annotated entities.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Training text must not be blank.")
        return stripped


class EntityPrediction(BaseModel):
    """NER extraction result."""

    text: str = Field(..., description="Extracted entity text.")
    label: str = Field(..., description="Predicted entity label.")
    start: int = Field(..., ge=0, description="Start character offset.")
    end: int = Field(..., gt=0, description="End character offset.")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score.")


class ExtractRequest(BaseModel):
    """Request body for NER extraction."""

    text: str = Field(..., min_length=1, max_length=5000, description="Input text.")
    model_version: str | None = Field(default=None, description="Optional explicit model version.")
    use_ab_test: bool = Field(default=True, description="Whether to route through the configured rollout.")
    audience_id: str | None = Field(default=None, description="Stable user or tenant id for deterministic A/B routing.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Input text must not be blank.")
        return stripped


class ExtractResponse(BaseModel):
    """Response body for NER extraction."""

    text: str = Field(..., description="Original request text.")
    model_version: str = Field(..., description="Model version used for extraction.")
    routing_strategy: str = Field(..., description="How the model version was selected.")
    entities: list[EntityPrediction] = Field(..., description="Detected entities.")


class TrainingJobRequest(BaseModel):
    """Request body for model training."""

    version_name: str = Field(..., min_length=3, max_length=64, description="New model version name.")
    base_version: str | None = Field(default=None, description="Base model version to extend.")
    examples: list[TrainingExample] = Field(..., min_length=1, description="Annotated examples.")
    auto_activate: bool = Field(default=False, description="Whether to activate the trained model after completion.")
    run_async: bool = Field(default=True, description="Whether to schedule the job asynchronously.")

    @field_validator("version_name")
    @classmethod
    def validate_version_name(cls, value: str) -> str:
        sanitized = value.strip()
        if not sanitized:
            raise ValueError("Version name must not be blank.")
        return sanitized


class TrainingJobResponse(BaseModel):
    """Initial response when creating a training job."""

    job_id: str = Field(..., description="Training job identifier.")
    status: str = Field(..., description="Current job status.")
    requested_version: str = Field(..., description="Requested model version.")


class TrainingJobStatusResponse(BaseModel):
    """Detailed training job status."""

    job_id: str = Field(..., description="Training job identifier.")
    requested_version: str = Field(..., description="Requested model version.")
    status: str = Field(..., description="queued, running, completed, or failed.")
    progress: int = Field(..., ge=0, le=100, description="Approximate completion percentage.")
    current_step: str = Field(..., description="Current step description.")
    created_version: str | None = Field(default=None, description="Created model version, when complete.")
    error: str | None = Field(default=None, description="Error details when failed.")


class RolloutConfigResponse(BaseModel):
    """Current A/B rollout configuration."""

    primary_version: str | None = Field(default=None, description="Primary production model version.")
    candidate_version: str | None = Field(default=None, description="Candidate model version.")
    candidate_percentage: float = Field(..., ge=0.0, le=1.0, description="Traffic percentage routed to the candidate.")


class RolloutConfigRequest(BaseModel):
    """Request body for updating A/B rollout settings."""

    primary_version: str = Field(..., min_length=1, description="Primary model version.")
    candidate_version: str = Field(..., min_length=1, description="Candidate model version.")
    candidate_percentage: float = Field(..., ge=0.0, le=1.0, description="Traffic percentage for candidate model.")


class ActivateModelRequest(BaseModel):
    """Request body for activating a model version."""

    version_name: str = Field(..., min_length=1, description="Model version to activate.")


class ModelVersionInfo(BaseModel):
    """Persisted model version metadata."""

    version_name: str = Field(..., description="Model version name.")
    backend: str = Field(..., description="Training/inference backend.")
    base_model_name: str = Field(..., description="Base transformer model name.")
    created_at: str = Field(..., description="UTC timestamp when the version was created.")
    training_example_count: int = Field(..., ge=0, description="Number of training examples used.")
    entity_labels: list[str] = Field(..., description="Labels covered by the version.")
    is_active: bool = Field(..., description="Whether this version is active.")


class ModelsResponse(BaseModel):
    """Collection of available model versions."""

    active_version: str = Field(..., description="Active serving version.")
    rollout: RolloutConfigResponse = Field(..., description="Current A/B rollout settings.")
    models: list[ModelVersionInfo] = Field(..., description="Available model versions.")


class HealthCheckResponse(BaseModel):
    """Health response with model and job status."""

    status: str = Field(..., description="Service health status.")
    active_version: str = Field(..., description="Currently active model version.")
    available_versions: int = Field(..., ge=1, description="Number of available model versions.")
    running_jobs: int = Field(..., ge=0, description="Number of running training jobs.")
    rollout: RolloutConfigResponse = Field(..., description="Current rollout settings.")