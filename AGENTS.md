# AGENTS.md: Development & Operational Rules for AI Agents

Welcome, AI Agent. When working on this repository (`identify-behavior`), you **MUST** adhere to the following architectural, performance, and coding constraints.

---

## 1. Non-Negotiable Performance Constraints

1. **Never Process 30 FPS for AI Inference:**
   * AI models must strictly execute at $\le 5\text{ FPS}$ per stream.
   * Frame ingestion and AI inference must remain decoupled via `LatestFrameStore`.

2. **CPU Threading Limits:**
   * Any execution touching `torch`, `numpy`, or `cv2` must respect:
     ```python
     os.environ["OMP_NUM_THREADS"] = "2"
     os.environ["MKL_NUM_THREADS"] = "2"
     os.environ["TORCH_NUM_THREADS"] = "2"
     torch.set_num_threads(2)
     ```
   * Never spawn unmetered worker threads. Use `ThreadPoolExecutor(max_workers=2)`.

3. **Memory Ceiling & State Bounding:**
   * Target memory footprint: $\le 300-400\text{ MB}$ per camera.
   * Maximum track history buffer is strictly capped at `MAX_TRACK_HISTORY = 15`.
   * Stale tracks (unseen for $> 5$ frames) must be purged immediately.

---

## 2. Architectural Invariants

* **Singleton Model Loading:** Never instantiate YOLO models inside camera worker loops or per-camera objects. All models must be accessed via `ModelLoader()`.
* **Zero Instant Snapshots:** Never persist snapshot images directly from a raw detection score. All snapshots must go through `SnapshotManager.save_if_confirmed()` and verify `EventState.is_confirmed()`.
* **High-Confidence Vehicle Filtering:** Discard vehicle bounding boxes with confidence $< 0.65$ before entering track creation to eliminate false tracks and memory churn.

---

## 3. Code Quality & Contribution Standards

* **Preserve Documentation:** Maintain docstrings, type hints, and existing comments across refactoring sessions.
* **Test Before Submitting:** Run unit tests in `tests/` before completing tasks.
* **Keep Hand-off Logs Synchronized:** Keep `HANDOFF.md`, `TASKS.md`, and `CHANGELOG.md` updated as features transition through P0, P1, and P2 stages.
