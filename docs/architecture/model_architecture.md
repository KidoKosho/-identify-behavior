# Model Architecture

## 1. System Invariants
- **Inference Decoupling:** AI inference runs asynchronously at a target rate of ≤ 5 FPS per stream.
- **Latest-Frame Storage:** Frame ingestion writes exclusively to `LatestFrameStore`. Old frames are overwritten; zero frame queues.
- **Singleton Model Loading:** Models are loaded exactly once per process via `ModelRegistry`.
- **CPU Thread Bounds:** Maximum 2 worker threads for OpenCV/PyTorch/OMP/MKL.
- **State-Driven Confirmation:** Raw detections are Candidates. Events are confirmed across a rolling temporal window (N ≥ 3 frames).
- **Bounded History:** Maximum track history is strictly capped at `MAX_TRACK_HISTORY = 15`. Stale tracks are purged.

## 2. Component Boundaries

### Camera Layer
- `CameraWorker`: Dedicated non-blocking capture thread.
- `LatestFrameStore`: Single-slot per camera thread-safe store.

### Scheduling & Execution
- `InferenceScheduler`: Throttles inference to 5 FPS.

### Core Processing
- `ModelRegistry`: Centralized loading of YOLO and specialized models.
- `Detector` / `Pipeline`: Gating logic, filtering, and cascading rules.

### Output & Event Management
- `EventState`: FSM defining NORMAL → DETECTED → CANDIDATE → CONFIRMING → CONFIRMED.
- `SnapshotManager`: Handles snapshots strictly on confirmed events.

## 3. Data Contracts
- `Detection`, `Track`, `EventCandidate`, and `Event` are strictly typed via dataclasses in `temporal/contracts.py`.
