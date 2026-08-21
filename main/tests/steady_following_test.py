import unittest
from collections import deque

from scenarios.overtaking_cyclist_bidirectional_road import (
    is_steady_following,
    STABILITY_WINDOW,
    DISTANCE_STABLE_EPS,
    SPEED_CONVERGENCE_EPS,
)


class SteadyFollowingTestCase(unittest.TestCase):

    def test_insufficient_history_returns_false(self):
        # Fewer than STABILITY_WINDOW samples — not enough evidence to call it steady yet.
        history = deque([10.0, 9.9], maxlen=STABILITY_WINDOW)
        self.assertFalse(is_steady_following(history, ego_speed=5.0, obstacle_speed=5.0))

    def test_steady_case_distance_flat_speeds_converged(self):
        # Distance essentially constant, speeds matched -> steady following.
        history = deque([10.0, 10.01, 9.98, 10.02, 9.99], maxlen=STABILITY_WINDOW)
        self.assertTrue(is_steady_following(history, ego_speed=5.0, obstacle_speed=5.1))

    def test_closing_case_distance_decreasing(self):
        # Distance clearly shrinking each frame -> still actively closing, not steady.
        history = deque([10.0, 8.0, 6.0, 4.0, 2.0], maxlen=STABILITY_WINDOW)
        self.assertFalse(is_steady_following(history, ego_speed=5.0, obstacle_speed=5.0))

    def test_closing_case_speeds_not_converged(self):
        # Distance flat but ego still much faster than obstacle -> not yet steady.
        history = deque([10.0, 10.01, 9.98, 10.02, 9.99], maxlen=STABILITY_WINDOW)
        self.assertFalse(is_steady_following(history, ego_speed=8.0, obstacle_speed=5.0))


if __name__ == '__main__':
    unittest.main()
