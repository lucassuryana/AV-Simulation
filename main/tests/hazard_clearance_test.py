import math
import unittest
from collections import deque

from scenarios.overtaking_cyclist_bidirectional_road import (
    is_opposing_traffic,
    is_hazard_cleared,
    STABILITY_WINDOW,
    DISTANCE_STABLE_EPS,
)


class IsOpposingTrafficTestCase(unittest.TestCase):

    def test_same_direction_is_not_opposing(self):
        # Cyclist and ego both heading "up" the road.
        self.assertFalse(is_opposing_traffic(obstacle_heading=math.pi / 2, ego_heading=math.pi / 2))

    def test_slightly_different_heading_is_not_opposing(self):
        # Ego mid-steer during an overtake — still broadly the same direction.
        self.assertFalse(is_opposing_traffic(obstacle_heading=math.pi / 2, ego_heading=math.pi / 2 - 0.3))

    def test_opposite_heading_is_opposing(self):
        # Oncoming car heading "down" while the ego heads "up".
        self.assertTrue(is_opposing_traffic(obstacle_heading=-math.pi / 2, ego_heading=math.pi / 2))

    def test_wraparound_does_not_falsely_report_opposing(self):
        # Both headings point almost the same direction (near +-pi), but are expressed on
        # opposite sides of the wraparound boundary -- a naive abs(a - b) would see ~6.18
        # rad of difference and misclassify this as opposing. The actual angular gap is
        # only ~0.1 rad, so this must NOT read as opposing.
        self.assertFalse(is_opposing_traffic(obstacle_heading=-(math.pi - 0.05), ego_heading=math.pi - 0.05))

    def test_perpendicular_is_exactly_the_threshold_not_opposing(self):
        # angle_threshold is exclusive (diff > threshold), so exactly perpendicular
        # (diff == pi/2) counts as not-opposing.
        self.assertFalse(is_opposing_traffic(obstacle_heading=0.0, ego_heading=math.pi / 2))


class IsHazardClearedTestCase(unittest.TestCase):

    def test_insufficient_history_returns_false(self):
        history = deque([10.0, 9.9], maxlen=STABILITY_WINDOW)
        self.assertFalse(is_hazard_cleared(history))

    def test_distance_increasing_is_cleared(self):
        # Oncoming car has passed and distance is opening back up.
        history = deque([2.0, 2.5, 3.2, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], maxlen=STABILITY_WINDOW)
        self.assertTrue(is_hazard_cleared(history))

    def test_distance_still_closing_is_not_cleared(self):
        history = deque([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], maxlen=STABILITY_WINDOW)
        self.assertFalse(is_hazard_cleared(history))

    def test_does_not_require_speed_convergence(self):
        # Ego stopped (v=0) while the obstacle still moves away at full speed in the
        # opposite direction -- speeds will never "converge", but the hazard has still
        # cleared since distance is opening. This is exactly the case is_steady_following
        # gets wrong for opposing traffic.
        history = deque([1.5, 2.0, 2.8, 3.9, 5.3, 7.0, 9.0, 11.3, 13.9, 16.8], maxlen=STABILITY_WINDOW)
        self.assertTrue(is_hazard_cleared(history))

    def test_flat_distance_within_eps_is_cleared(self):
        history = deque([10.0, 10.01, 9.98, 10.02, 9.99, 10.0, 10.01, 9.98, 10.02, 9.99], maxlen=STABILITY_WINDOW)
        self.assertTrue(is_hazard_cleared(history))


if __name__ == '__main__':
    unittest.main()
