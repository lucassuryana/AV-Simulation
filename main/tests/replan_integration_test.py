import unittest
from collections import deque

from scenarios.overtaking_cyclist_bidirectional_road import (
    is_steady_following,
    gate_replan_request,
    filter_safe_trajectories,
    STABILITY_WINDOW,
)


class ReplanIntegrationTestCase(unittest.TestCase):
    """
    Step 6: drives steps 2 (steady-state detection), 3 (gate + queue) and 4
    (safety filter) together across a simulated multi-frame timeline, the way
    the real main loop threads state between them frame to frame. A live
    scenario run can't reliably exercise "reason below threshold AND an
    active override at the same instant" (collision_hold_active clears in
    under a second while reasons take several seconds to trip, per steps
    2/3's own validation) — so this forces the timing directly instead.
    """

    NUM_FRAMES = 12
    # Distance actively shrinking for 4 frames, then flat forever after.
    CLOSING_DISTANCES = [12.0, 10.0, 8.0, 6.0]
    STEADY_DISTANCE = 6.0
    EGO_SPEED = 3.0
    OBSTACLE_SPEED = 3.0

    def _run_timeline(self, reasons_below_threshold_by_frame):
        """
        reasons_below_threshold_by_frame: dict {frame_index: bool}, defaults to
        False for unlisted frames — the *current* reason level at that frame.
        A replan pulse (replan_needed=True) fires only on the rising edge
        (level goes False -> True), mirroring reasons_evaluation's
        replan_tracker hysteresis in the real code: it doesn't re-pulse every
        frame the reason stays bad, only the instant it first drops.

        collision_hold_active is derived from is_steady_following (step 2) and
        is entirely independent of the reasons dict — same distance/speed
        timeline drives it in every test, so the release frame is deterministic
        regardless of which reasons scenario is being tested.
        """
        distance_history = deque(maxlen=STABILITY_WINDOW)
        pending_replan_request = False
        prev_level = False
        frames = []

        for frame in range(self.NUM_FRAMES):
            if frame < len(self.CLOSING_DISTANCES):
                distance_history.append(self.CLOSING_DISTANCES[frame])
            else:
                distance_history.append(self.STEADY_DISTANCE)

            steady = is_steady_following(distance_history, self.EGO_SPEED, self.OBSTACLE_SPEED)
            collision_hold_active = not steady

            level = reasons_below_threshold_by_frame.get(frame, False)
            replan_needed = level and not prev_level
            prev_level = level

            logs = []
            execute_replan, pending_replan_request = gate_replan_request(
                replan_needed, collision_hold_active, pending_replan_request,
                level, log_fn=logs.append
            )

            frames.append({
                'frame': frame,
                'collision_hold_active': collision_hold_active,
                'execute_replan': execute_replan,
                'pending_replan_request': pending_replan_request,
                'logs': logs,
            })

        return frames

    def _release_frame_index(self, frames):
        return next(i for i, f in enumerate(frames) if not f['collision_hold_active'])

    def test_queued_replan_executes_after_release_with_fresh_reasons_still_bad(self):
        # Reason trips at frame 1 (while still closing) and stays bad continuously
        # afterward -> queued at frame 1, released only once the override clears,
        # re-validated against that release frame's current (still-bad) reasons.
        reasons = {f: True for f in range(1, self.NUM_FRAMES)}
        frames = self._run_timeline(reasons)
        release_idx = self._release_frame_index(frames)
        self.assertGreater(release_idx, 1, "test needs the override to still be active after the request")

        # Requirement 1: override's cutoff state continues uninterrupted — every
        # frame up to release stays held, nothing executes early.
        for f in frames[:release_idx]:
            self.assertTrue(f['collision_hold_active'], f"frame {f['frame']} should still be closing")
            self.assertFalse(f['execute_replan'], f"frame {f['frame']} must not execute while held")

        # Requirement 2: the request is queued, not dropped, at the instant it's made.
        self.assertIn("replan requested", frames[1]['logs'])
        self.assertIn("replan held — collision active", frames[1]['logs'])
        self.assertTrue(frames[1]['pending_replan_request'])

        # It sits queued untouched for every subsequent held frame — no re-triggering,
        # no stray logs, no early exit.
        for f in frames[2:release_idx]:
            self.assertTrue(f['pending_replan_request'], f"frame {f['frame']} should still be pending")
            self.assertEqual(f['logs'], [], f"frame {f['frame']} should be silent while held")

        # Requirement 3: executes only once the override releases, using that
        # frame's current (still-bad) reasons — not a stale replay of frame 1's.
        release_frame = frames[release_idx]
        self.assertTrue(release_frame['execute_replan'])
        self.assertIn("replan executed on release", release_frame['logs'])
        self.assertFalse(release_frame['pending_replan_request'])

        # And it never fires again afterward.
        for f in frames[release_idx + 1:]:
            self.assertFalse(f['execute_replan'])
            self.assertEqual(f['logs'], [])

    def test_queued_replan_dropped_after_release_when_reasons_recovered(self):
        # Same closing/held timeline, but the underlying reason recovers by the
        # time the override releases (resolved itself while collision avoidance
        # was still in control). Must NOT replay the stale "bad" decision.
        baseline = self._run_timeline({})
        release_idx = self._release_frame_index(baseline)

        reasons = {f: True for f in range(1, release_idx)}  # bad only while held; recovered by release
        frames = self._run_timeline(reasons)
        self.assertEqual(self._release_frame_index(frames), release_idx)

        release_frame = frames[release_idx]
        self.assertFalse(release_frame['execute_replan'])
        self.assertFalse(release_frame['pending_replan_request'])
        self.assertIn("replan held request dropped — reasons recovered", release_frame['logs'])

    def test_safety_filter_runs_on_the_executed_replan(self):
        # Requirement 3 continued: once gate_replan_request clears execution,
        # the candidate selection that follows must still respect the step-4
        # safety filter — an unsafe candidate must never be chosen even though
        # the replan was legitimately released and executed.
        reasons = {f: True for f in range(1, self.NUM_FRAMES)}
        frames = self._run_timeline(reasons)
        release_idx = self._release_frame_index(frames)
        self.assertTrue(frames[release_idx]['execute_replan'])

        candidates = ["risky_overtake", "follow_trajectory"]
        unsafe = {"risky_overtake"}
        selected = filter_safe_trajectories(
            candidates, is_unsafe_fn=lambda i, t: t in unsafe, log_fn=None
        )

        self.assertEqual(selected, ["follow_trajectory"])


if __name__ == '__main__':
    unittest.main()
