# Project 9 - PyTorch Fine-Tuning and Training Pipeline

> A design brief for the missing project in this roadmap. This project fills the gap between consuming pretrained models and operating production AI systems by teaching you how to train, fine-tune, evaluate, and package a model with PyTorch in an AI engineering context.

---

## 1. What this project should do

Build a small but production-shaped training system that:

1. Loads labeled text data from disk.
2. Preprocesses and tokenizes it for a transformer model.
3. Fine-tunes a pretrained HuggingFace model using **PyTorch**.
4. Tracks training and validation metrics across epochs.
5. Saves checkpoints and the best-performing model artifact.
6. Exposes the trained model through a FastAPI inference service.
7. Separates **offline training** from **online inference** so the deployment boundary is explicit.

The point is not "do deep learning theory." The point is to learn the operational shape of model training as an AI engineer.

---

## 2. Why this project belongs in the roadmap

Right now the roadmap teaches you how to:

- train a classical ML model
- consume pretrained NLP models
- build RAG systems
- manage model lifecycle and rollout
- build agents
- serve LLMs

What is missing is the step where you personally own a neural model's training lifecycle.

Without that step, you skip several core AI engineering skills:

- using `Dataset` and `DataLoader`
- writing a PyTorch training loop
- moving tensors and models onto CPU or GPU correctly
- checkpointing and resuming training jobs
- comparing train vs validation metrics
- packaging the trained weights for inference
- understanding the boundary between experimentation and service deployment

This project closes that gap cleanly.

---

## 3. The recommended use case

Use **text classification** as the task. It keeps the training pipeline understandable while still teaching the real mechanics.

Recommended problem:

- **Support ticket triage**

Example labels:

- `billing`
- `technical_issue`
- `account_access`
- `feature_request`
- `shipping`

Why this is a good AI engineering project:

- The input is realistic business text.
- The output is easy to validate.
- The model can be trained on a laptop GPU or CPU with a small dataset.
- The serving story is straightforward: a FastAPI endpoint returns the predicted class and confidence.
- It mirrors real production workflows better than an academic vision dataset.

---

## 4. Core learning goals

By the end of the project, you should understand:

### 4.1 The training boundary

Training is an **offline pipeline**, not something your API should do on each request.

You should produce an artifact like:

- model weights
- tokenizer files
- label mapping
- training metrics summary
- config used for training

Then your inference service should load those files at startup and serve predictions.

### 4.2 PyTorch fundamentals in production form

You should touch the pieces AI engineers actually use:

- `torch.utils.data.Dataset`
- `DataLoader`
- `model.train()` and `model.eval()`
- `torch.no_grad()`
- optimizer setup
- loss computation
- backpropagation with `loss.backward()`
- optimizer stepping and gradient reset
- checkpoint saving with `torch.save(...)`
- loading state with `load_state_dict(...)`

### 4.3 Device handling

You should explicitly support:

- CPU training
- GPU training when CUDA is available

That means moving:

- model parameters to `device`
- input tensors to `device`
- labels to `device`

This is basic, but it is the first place many engineers break a PyTorch project.

### 4.4 Evaluation discipline

The project should report at least:

- training loss
- validation loss
- validation accuracy
- macro F1 score

Why macro F1 matters: if your classes are imbalanced, accuracy can look fine while minority classes fail completely.

### 4.5 Artifact management

The training job should write artifacts to a versioned folder such as:

```text
models/
  ticket_classifier_v1/
    model.pt
    tokenizer/
    labels.json
    metrics.json
    training_config.json
```

This makes the later model lifecycle projects feel like a natural continuation.

---

## 5. Suggested project architecture

```text
Project_9/
  README.md
  requirements.txt
  app/
    __init__.py
    config.py
    dependencies.py
    main.py
    models/
    routers/
    services/
  data/
    raw/
    processed/
  models/
  scripts/
    prepare_dataset.py
    train_model.py
    evaluate_model.py
  tests/
    __init__.py
    test_api_endpoints.py
    test_training_pipeline.py
    test_inference_service.py
```

Recommended responsibilities:

- `scripts/prepare_dataset.py`: clean and split raw data into train, validation, and test sets
- `scripts/train_model.py`: run the PyTorch fine-tuning job
- `scripts/evaluate_model.py`: load saved artifacts and report final metrics
- `app/services/inference_service.py`: load the trained artifact and run predictions
- `app/routers/predictions.py`: expose `/predict` and `/health`
- `app/config.py`: centralize hyperparameters and paths

---

## 6. The end-to-end flow

```text
Raw CSV or JSON dataset
        |
        v
prepare_dataset.py
        |
        v
train_model.py
  - load tokenizer
  - build Dataset/DataLoader
  - fine-tune transformer with PyTorch
  - validate each epoch
  - save best checkpoint
        |
        v
models/ticket_classifier_v1/
        |
        v
FastAPI inference service
  - load artifact at startup
  - accept text input
  - return label + confidence
```

This is the operational boundary you want to internalize: scripts produce artifacts; services consume artifacts.

---

## 7. Recommended technical stack

- `torch`
- `transformers`
- `datasets` or a lightweight CSV/JSON loader
- `scikit-learn` for metrics
- `fastapi`
- `uvicorn`
- `pydantic`
- `pytest`

Recommended base model:

- `distilbert-base-uncased`

Why:

- small enough for a local machine
- common enough that the ecosystem examples are strong
- realistic enough to teach transfer learning and tokenizer management

---

## 8. What the training loop should teach

At minimum, the implementation should make these mechanics visible:

1. Load batches from a `DataLoader`.
2. Tokenize text into tensors.
3. Run a forward pass.
4. Compute classification loss.
5. Backpropagate gradients.
6. Step the optimizer.
7. Evaluate on validation data with gradients disabled.
8. Save the best checkpoint based on validation performance.

You do not need distributed training, mixed precision, LoRA, or DeepSpeed in the first version. Those are follow-ups, not prerequisites.

---

## 9. API contract

The online service can stay small.

### `POST /predict`

Request:

```json
{
  "text": "I was charged twice for my subscription this month."
}
```

Response:

```json
{
  "label": "billing",
  "confidence": 0.94,
  "model_version": "ticket_classifier_v1"
}
```

### `GET /health`

Should report:

- whether the model artifact loaded successfully
- active model version
- device in use (`cpu` or `cuda`)

---

## 10. Key engineering concepts this project should force you to learn

### 10.1 Why batch size matters

Larger batches improve throughput but consume more memory. On a GPU, this is often the first hyperparameter constrained by reality rather than theory.

### 10.2 Why validation must be separate from training

If you only report training loss, you can mistake memorization for improvement. Validation is your guardrail against overfitting.

### 10.3 Why checkpoints matter

Training jobs fail. Machines reboot. Bad experiments happen. Checkpoints let you:

- resume interrupted runs
- compare versions
- deploy the best model rather than the last epoch blindly

### 10.4 Why inference code should not depend on training-only objects

Your API should load a finished artifact and perform inference. It should not import your training script or carry training state around. Clean boundaries matter.

### 10.5 Why metrics belong in files, not only in logs

If the best validation F1 is only visible in terminal output, you have not really captured the result. Write metrics to JSON so later services and tests can inspect them.

---

## 11. Suggested milestones

### Milestone 1

Prepare a labeled dataset and split it into train, validation, and test.

### Milestone 2

Implement a PyTorch fine-tuning script that trains a classifier end to end.

### Milestone 3

Save a versioned artifact folder with weights, tokenizer, labels, and metrics.

### Milestone 4

Create a FastAPI service that loads the artifact once at startup and serves predictions.

### Milestone 5

Write tests that validate:

- artifact files exist after training
- model can be loaded for inference
- `/predict` returns a valid label and confidence
- `/health` reports degraded status if the artifact is missing

---

## 12. Stretch goals

Once the base version works, the next valuable upgrades are:

1. Early stopping based on validation loss.
2. Learning-rate scheduling.
3. Resume training from checkpoint.
4. TensorBoard or MLflow logging.
5. Class-weighted loss for imbalanced labels.
6. Export to ONNX or TorchScript for faster inference.
7. Dockerize the inference service.
8. Add a simple background training job API, then connect it conceptually to the model lifecycle work from the NER project.

---

## 13. Where this project fits in the roadmap

This project belongs before model lifecycle and before custom serving.

A cleaner sequence is:

1. Classical ML baseline
2. Pretrained transformer inference
3. Basic RAG
4. **PyTorch fine-tuning and training pipeline**
5. Model lifecycle and registry
6. Multi-modal AI
7. Agent with tool use
8. Vector database and advanced RAG
9. Custom LLM deployment and optimization

That order is better because it teaches:

- use a model
- adapt a model
- operate a model
- combine models into systems
- serve models at scale

---

## 14. The real lesson

The most important thing this project teaches is not just PyTorch syntax.

It teaches the difference between:

- a notebook experiment
- a repeatable training pipeline
- a deployable model artifact
- a production inference service

That difference is the center of AI engineering.