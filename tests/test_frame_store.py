import unittest
import numpy as np
import time
from app.camera.latest_frame_store import LatestFrameStore


class TestLatestFrameStore(unittest.TestCase):
    def setUp(self):
        # Reset store before test
        with LatestFrameStore._lock:
            LatestFrameStore._frames.clear()

    def test_single_slot_and_overwrite(self):
        cam_id = "cam_test"
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8)

        # Store frame 1
        LatestFrameStore.set(cam_id, frame1, 1)
        res1 = LatestFrameStore.get(cam_id)
        self.assertIsNotNone(res1)
        self.assertEqual(res1[1], 1)
        self.assertEqual(res1[0][0, 0, 0], 0)

        # Store frame 2 (should overwrite frame 1 in the same slot)
        LatestFrameStore.set(cam_id, frame2, 2)
        res2 = LatestFrameStore.get(cam_id)
        self.assertIsNotNone(res2)
        self.assertEqual(res2[1], 2)
        self.assertEqual(res2[0][0, 0, 0], 1)

        # Verify only 1 slot exists for this camera
        with LatestFrameStore._lock:
            self.assertEqual(len(LatestFrameStore._frames), 1)

    def test_multiple_cameras_isolated(self):
        f_a = np.zeros((10, 10, 3), dtype=np.uint8)
        f_b = np.ones((10, 10, 3), dtype=np.uint8)

        LatestFrameStore.set("cam_a", f_a, 100)
        LatestFrameStore.set("cam_b", f_b, 200)

        res_a = LatestFrameStore.get("cam_a")
        res_b = LatestFrameStore.get("cam_b")

        self.assertEqual(res_a[1], 100)
        self.assertEqual(res_b[1], 200)
        with LatestFrameStore._lock:
            self.assertEqual(len(LatestFrameStore._frames), 2)


if __name__ == "__main__":
    unittest.main()
