import unittest
from types import SimpleNamespace

from detector import Detector


class TrackingAssignmentTest(unittest.TestCase):
    def test_track_and_compute_velocity_keeps_unique_ids_for_nearby_boxes(self):
        detector = Detector(SimpleNamespace(coco_model=None, fire_model=None, accident_model=None))
        detector.prev_positions = {7: (100.0, 100.0)}
        detector.next_id = 8

        velocity_map, frame_records = detector.track_and_compute_velocity([
            (99.0, 99.0, 110.0, 110.0),
            (101.0, 101.0, 112.0, 112.0),
        ], frame_id=5)

        self.assertEqual(len(frame_records), 2)
        track_ids = [record["track_id"] for record in frame_records]
        self.assertEqual(len(set(track_ids)), 2)
        self.assertIn(7, track_ids)
        self.assertIn(8, track_ids)
        self.assertIn(7, velocity_map)
        self.assertIn(8, velocity_map)


if __name__ == "__main__":
    unittest.main()
