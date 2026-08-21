import unittest

from scenarios.overtaking_cyclist_bidirectional_road import (
    should_retry_after_fallback,
    is_trajectory_nearly_exhausted,
    is_near_true_goal,
)


class ShouldRetryAfterFallbackTestCase(unittest.TestCase):

    def test_no_retry_when_last_replan_was_not_a_fallback(self):
        # Last replan freely picked its preferred candidate — nothing to retry.
        self.assertFalse(should_retry_after_fallback(
            last_replan_was_fallback=False, reasons_below_threshold=True,
            time_since_last_replan=100.0, retry_interval=2.0
        ))

    def test_no_retry_when_reasons_have_recovered(self):
        # Was constrained by safety, but the underlying reason is fine now —
        # nothing left to fix by retrying.
        self.assertFalse(should_retry_after_fallback(
            last_replan_was_fallback=True, reasons_below_threshold=False,
            time_since_last_replan=100.0, retry_interval=2.0
        ))

    def test_no_retry_before_cooldown_elapses(self):
        self.assertFalse(should_retry_after_fallback(
            last_replan_was_fallback=True, reasons_below_threshold=True,
            time_since_last_replan=1.0, retry_interval=2.0
        ))

    def test_retry_once_cooldown_elapses_and_still_needed(self):
        self.assertTrue(should_retry_after_fallback(
            last_replan_was_fallback=True, reasons_below_threshold=True,
            time_since_last_replan=2.0, retry_interval=2.0
        ))

    def test_retry_well_after_cooldown(self):
        self.assertTrue(should_retry_after_fallback(
            last_replan_was_fallback=True, reasons_below_threshold=True,
            time_since_last_replan=50.0, retry_interval=2.0
        ))


class IsTrajectoryNearlyExhaustedTestCase(unittest.TestCase):

    def test_false_with_plenty_of_trajectory_remaining(self):
        self.assertFalse(is_trajectory_nearly_exhausted(traj_agent_idx=0, trajectory_len=100, margin=10))

    def test_true_right_at_the_margin(self):
        # Exactly `margin` points remaining -> should trigger (boundary is inclusive).
        self.assertTrue(is_trajectory_nearly_exhausted(traj_agent_idx=90, trajectory_len=100, margin=10))

    def test_true_with_fewer_than_margin_remaining(self):
        self.assertTrue(is_trajectory_nearly_exhausted(traj_agent_idx=95, trajectory_len=100, margin=10))

    def test_true_when_already_at_the_last_point(self):
        self.assertTrue(is_trajectory_nearly_exhausted(traj_agent_idx=99, trajectory_len=100, margin=10))

    def test_false_just_outside_the_margin(self):
        self.assertFalse(is_trajectory_nearly_exhausted(traj_agent_idx=89, trajectory_len=100, margin=10))


class IsNearTrueGoalTestCase(unittest.TestCase):

    def test_false_when_far_away(self):
        self.assertFalse(is_near_true_goal(0.0, 0.0, 2.0, 22.0, margin=5.0))

    def test_true_right_at_the_goal(self):
        self.assertTrue(is_near_true_goal(2.0, 22.0, 2.0, 22.0, margin=5.0))

    def test_true_within_margin(self):
        # ~1.81m away, matching the real crash scenario that motivated this check.
        self.assertTrue(is_near_true_goal(1.38, 20.30, 2.0, 22.0, margin=5.0))

    def test_false_just_outside_margin(self):
        self.assertFalse(is_near_true_goal(2.0, 27.1, 2.0, 22.0, margin=5.0))

    def test_true_just_inside_margin(self):
        self.assertTrue(is_near_true_goal(2.0, 26.9, 2.0, 22.0, margin=5.0))


if __name__ == '__main__':
    unittest.main()
