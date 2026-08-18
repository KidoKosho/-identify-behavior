# AUDIT: identify-behavior System

**Date:** 2026-08-17  
**Role:** Senior Computer Vision Engineer & Software Architect  
**Repository:** https://github.com/KidoKosho/-identify-behavior  

---

## 1. Executive System Inspection (15 Key Points)

| # | Architecture Component | Current Code Location | Implementation Details & Status |
|---|---|---|---|
| **1** | **Application Entry Point** | [`main.py`](file:///e:/Code/python/test/main.py), [`run_demo.py`](file:///e:/Code/python/test/run_demo.py), [`scheduler/inference_scheduler.py`](file:///e:/Code/python/test/scheduler/inference_scheduler.py) | CLI argument parser (`--video`, `--output`, `--display`, `--window-scale`), sets CPU thread limits, initializes `ModelLoader` & `Pipeline`. |
| **2** | **Camera Manager & Worker** | [`camera/camera_worker.py`](file:///e:/Code/python/test/camera/camera_worker.py) | `CameraWorker` captures frames in daemon thread asynchronously, writes only latest frame to `LatestFrameStore`. |
| **3** | **ModelLoader (Loading Location)** | [`models/model_loader.py`](file:///e:/Code/python/test/models/model_loader.py), [`models.py`](file:///e:/Code/python/test/models.py) | `models/model_loader.py` implements Singleton pattern (`__new__`) for process-wide shared instances. `models.py` serves as legacy wrapper. |
| **4** | **YOLO (COCO) Invocation** | [`detector.py`](file:///e:/Code/python/test/detector.py): `Detector.detect_objects()` | Invokes `coco_model(frame, imgsz=640, conf=0.22)`. Vehicle confidence filter (`>= 0.65`) active. |
| **5** | **Fire/Smoke Model Invocation** | [`detector.py`](file:///e:/Code/python/test/detector.py): `Detector.detect_smoke()` | Invokes `fire_model(frame, imgsz=224, conf=0.30)`. Gated by fast color heuristic (`_has_fire_smoke_candidate`). |
| **6** | **Violence / Fight Detection** | [`detector.py`](file:///e:/Code/python/test/detector.py): `Detector.detect_fight()` | Uses multi-factor heuristic (persons $\ge 2$, IoU > 0.58, velocity > 10). No neural network violence model loaded currently. |
| **7** | **Embedding / MobileNet / FAISS** | Not present in active pipeline (Planned P2) | `tf_lite_model.tflite` exists in root as standalone artifact. Async embedding worker planned for P2-T01. |
| **8** | **Tracking & Track History** | [`detector.py`](file:///e:/Code/python/test/detector.py): `Detector.track_and_compute_velocity()` | Centroid distance tracking with Hungarian-like greedy match. History bounded to `MAX_TRACK_HISTORY = 15`. Purges stale tracks $> 5$ frames. Memory records capped at 1000 items. |
| **9** | **Snapshot Logic** | [`snapshot/snapshot_manager.py`](file:///e:/Code/python/test/snapshot/snapshot_manager.py), [`pipeline.py`](file:///e:/Code/python/test/pipeline.py) | `SnapshotManager.save_if_confirmed()` writes JPEG image with 10s cooldown per camera. |
| **10** | **Event & Temporal State Logic** | [`temporal/event_state.py`](file:///e:/Code/python/test/temporal/event_state.py) | `EventState` manages Candidate $\to$ Confirmed state machine with confirmation threshold ($N=3$) and cooldown expiration. |
| **11** | **Queue & Frame Storage** | [`camera/latest_frame_store.py`](file:///e:/Code/python/test/camera/latest_frame_store.py) | Single-slot thread-safe dictionary per camera (`_frames[cam_id] = (frame, count, ts)`). Zero unbounded queueing. |
| **12** | **PyTorch / OpenMP Thread Limits** | [`config/performance_config.py`](file:///e:/Code/python/test/config/performance_config.py), [`main.py`](file:///e:/Code/python/test/main.py) | Enforces `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `TORCH_NUM_THREADS=2`, `torch.set_num_threads(2)`. |
| **13** | **Existing Test Suite** | [`tests/`](file:///e:/Code/python/test/tests/) | 9 unit tests passing: `test_tracking_assignment.py` (3 tests), `test_vehicle_filter.py` (2 tests), `test_cascade.py` (4 tests). |
| **14** | **Configuration Systems** | [`config.py`](file:///e:/Code/python/test/config.py), [`config/performance_config.py`](file:///e:/Code/python/test/config/performance_config.py), [`config/model_config.py`](file:///e:/Code/python/test/config/model_config.py) | Global parameters, thresholds, paths, performance limits centralized. |
| **15** | **Dependencies & Execution** | [`requirements.txt`](file:///e:/Code/python/test/requirements.txt) | Python 3.11 with `ultralytics`, `opencv-python`, `torch`, `torchvision`, `numpy`, `pandas`. Run with `python main.py --video <path>`. |

---

## 2. Comprehensive Audit & Refactoring Matrix

| Feedback | File | Function/Class | Current behavior | Problem | Proposed fix | Phase |
|---|---|---|---|---|---|---|
| **CPU Thread Contention** | `main.py`, `models/model_loader.py` | Global startup | Thread limits partially applied across multiple entry points. | If models are imported before `os.environ` or in sub-threads, default thread pools may allocate all CPU cores. | Centralize runtime thread clamping in `config/performance_config.py` executed before any heavy imports. | **P0-T02** |
| **Model Duplication Risk** | `models.py` vs `models/model_loader.py` | `ModelLoader` | Two separate `ModelLoader` classes exist in the repository (`models.py` vs `models/model_loader.py`). | Legacy `models.py` creates new YOLO instances on every instantiation, risking memory duplication. | Unify `models.py` to alias or wrap the Singleton `ModelRegistry` in `models/model_loader.py` with instance verification logs. | **P0-T03** |
| **Frame Accumulation & Ingestion** | `camera/latest_frame_store.py` | `LatestFrameStore` | Stores 1 frame per camera under thread lock. | No explicit eviction policy or dropped frame counter for metrics tracking. | Add capture metrics (ingested FPS, dropped frames, lock latency) to `LatestFrameStore`. | **P0-T04** |
| **Camera Worker Decoupling** | `camera/camera_worker.py` | `CameraWorker._capture_loop` | Runs OpenCV capture in background thread and writes to `LatestFrameStore`. | Does not handle stream reconnection with exponential backoff on network drop. | Add robust auto-reconnection loop and capture-only verification tests. | **P0-T05** |
| **Inference Scheduling** | `scheduler/inference_scheduler.py` | `InferenceScheduler.run` | Polls `LatestFrameStore` at 5 FPS using `time.sleep(0.01)`. | Does not provide per-camera inference FPS logging or overload drop policy. | Implement timestamp-based token bucket / interval scheduling with per-camera inference FPS telemetry. | **P0-T06** |
| **Detection Cascade Hierarchy** | `detector.py` | `detect_smoke`, `detect_accident` | Level 0/1/2/3 cascade gates partially implemented. | Gating rules need formalization into multi-level cascade pipeline (Level 0 $\to$ Level 5). | Formally structure Level 0 (pre-filter) $\to$ Level 1 (YOLO) $\to$ Level 2 (geometry/color) $\to$ Level 3 (specialized) $\to$ Level 4 (temporal) $\to$ Level 5 (confirmed). | **P0-T07** |
| **Tracking History Bounding** | `detector.py` | `track_and_compute_velocity` | History is bounded to `MAX_TRACK_HISTORY = 15` and memory records capped at 1000. | In-memory `tracking_records` uses list slicing instead of `collections.deque`. | Refactor track history and records to `collections.deque(maxlen=15)` and `deque(maxlen=1000)` for $O(1)$ sliding window. | **P0-T08** |
| **Candidate Snapshot Ban** | `pipeline.py` | `process_frame` | Events routed through `EventState.add_candidate()`. | If `add_candidate` triggers, immediate snapshot is saved without multi-frame confirmation in some branches. | Strictly disallow any candidate snapshots; only trigger when state equals `EventState.CONFIRMED`. | **P0-T09** |
| **Multi-Stream Benchmarking** | `tests/` | Benchmark suite | Only unit tests exist; no formal synthetic 1-cam / multi-cam benchmark harness. | Cannot prove $\le 25\%$ CPU and $\le 400\text{ MB}$ RAM compliance under load. | Implement `tests/benchmark_p0.py` measuring CPU avg/peak, RAM delta, inference FPS, and dropped frames. | **P0-T10** |
| **State Machine Granularity** | `temporal/event_state.py` | `EventState` | Binary confirmed boolean (`candidate_count`, `confirmed`). | Lacks explicit state transitions (`NORMAL` $\to$ `DETECTED` $\to$ `CANDIDATE` $\to$ `CONFIRMING` $\to$ `CONFIRMED` $\to$ `COOLDOWN`). | Implement formal finite state machine (FSM) enum with `REJECTED` transition for non-persistent candidates. | **P1-T01** |
| **Fight Detection Heuristics** | `detector.py` | `detect_fight` | Uses single-frame IoU and velocity threshold. | Causes false positives on people walking close together or briskly passing by. | Implement multi-factor scoring (proximity + contact IoU + relative motion + motion intensity + temporal window). | **P1-T03** |
| **Vehicle Accident Heuristics** | `detector.py` | `detect_accident` | Calls `accident.pt` when vehicles present. | Lacks dual-track collision geometry and sudden deceleration delta verification. | Combine `accident.pt` confidence with trajectory vector intersection and deceleration delta. | **P1-T04** |
| **Fire & Smoke Persistence** | `detector.py` | `detect_smoke` | Color heuristic + model confidence. | Smoke model can trigger on gray dust or moving shadows without temporal consistency. | Enforce 3+ frame spatial-temporal persistence and bounding box stability before confirming smoke. | **P1-T05** |
