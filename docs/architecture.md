# ARCHITECTURE: Identify-Behavior System

## 1. System Invariants

1. **Inference Decoupling:** AI inference runs asynchronously at a target rate of $\le 5\text{ FPS}$ per stream.
2. **Latest-Frame Storage:** Frame ingestion via `CameraWorker` writes exclusively to `LatestFrameStore`. Old frames are overwritten; zero frame queues.
3. **Singleton Model Loading:** Models are loaded exactly once per process via `ModelRegistry` / `ModelLoader`.
4. **CPU Thread Bounds:** Maximum 2 worker threads for OpenCV/PyTorch/OMP/MKL (`OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `TORCH_NUM_THREADS=2`).
5. **State-Driven Confirmation:** Raw detections are Candidates. Only events confirmed across a rolling temporal window ($N \ge 3$ frames) trigger snapshots and alerts.
6. **Bounded History:** Maximum track history is strictly capped at `MAX_TRACK_HISTORY = 15`. Stale tracks ($> 5$ frames lost) are purged immediately.

---

## 2. Multi-Stage Cascade Architecture

```
[ Camera Stream (30 FPS) ]
            │
            ▼
┌───────────────────────────────┐
│   CameraWorker (Daemon)       │  Capture only, zero inference
└──────────────┬────────────────┘
               │ Overwrite latest slot
               ▼
┌───────────────────────────────┐
│     LatestFrameStore          │  Single-slot per camera (thread-safe)
└──────────────┬────────────────┘
               │ 5 FPS Polling (1.0 / INFERENCE_FPS)
               ▼
┌───────────────────────────────┐
│    InferenceScheduler         │  ThreadPoolExecutor(max_workers=2)
└──────────────┬────────────────┘
               │
   Level 0: Cheap Pre-filtering & Fast Color Analysis
               │
   Level 1: Shared YOLOv8 COCO (Person & Vehicle Detection)
               │ (Vehicle Conf >= 0.65 filter)
               ▼
┌─────────────────────────────────────────────────────────┐
│                   BEHAVIOR DETECTORS                    │
│                                                         │
│  Level 2: Candidate Extraction                          │
│  ├── Fight Candidate (Proximity & Relative Motion)      │
│  ├── Accident Candidate (Vehicle Closing Vector)        │
│  └── Fire / Smoke Candidate (Color & Brightness Anomaly)│
│                                                         │
│  Level 3: Specialized Model Execution                   │
│  ├── fire.pt (Only on fire/smoke candidate)             │
│  └── accident.pt (Only on vehicle interaction candidate)│
└──────────────┬──────────────────────────────────────────┘
               │
   Level 4: Temporal Confirmation (EventState State Machine)
               │ (NORMAL → DETECTED → CANDIDATE → CONFIRMING → CONFIRMED)
               ▼
┌───────────────────────────────┐
│     Confirmed Event           │  (Passed temporal stability check)
└──────────────┬────────────────┘
               │
   Level 5: Output & Persistence
               │
┌───────────────────────────────┐
│     SnapshotManager           │  (10s cooldown per camera)
└───────────────────────────────┘
```

---

## 3. Data Flow & Resource Budgets

| Resource | Target Budget | Mechanism |
|---|---|---|
| **CPU Usage** | $\le 20-25\%$ per camera | 5 FPS inference rate, 2 CPU threads |
| **RAM Footprint** | $\le 300-400\text{ MB}$ per camera | Singleton model loader, bounded track history ($M=15$) |
| **Snapshot Rate** | 0 snapshots on candidates | Confirmation required via `EventState.is_confirmed()` |
| **Inference Rate** | Target 5 FPS | Fixed-rate scheduling via `InferenceScheduler` |
