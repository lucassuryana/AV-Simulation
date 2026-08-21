import unittest

from scenarios.overtaking_cyclist_bidirectional_road import gate_replan_request


class ReplanGateTestCase(unittest.TestCase):

    def _call(self, replan_needed, collision_hold_active, pending_replan_request, reasons_below_threshold):
        logs = []
        execute_replan, pending_replan_request = gate_replan_request(
            replan_needed, collision_hold_active, pending_replan_request, reasons_below_threshold,
            log_fn=logs.append
        )
        return execute_replan, pending_replan_request, logs

    def test_fresh_request_executes_immediately_when_not_held(self):
        # No collision override active -> request goes straight through.
        execute_replan, pending, logs = self._call(
            replan_needed=True, collision_hold_active=False,
            pending_replan_request=False, reasons_below_threshold=True
        )
        self.assertTrue(execute_replan)
        self.assertFalse(pending)
        self.assertEqual(logs, ["replan requested"])

    def test_fresh_request_held_when_collision_active(self):
        # Collision avoidance is actively closing on an obstacle -> queue, don't execute.
        execute_replan, pending, logs = self._call(
            replan_needed=True, collision_hold_active=True,
            pending_replan_request=False, reasons_below_threshold=True
        )
        self.assertFalse(execute_replan)
        self.assertTrue(pending)
        self.assertEqual(logs, ["replan requested", "replan held — collision active"])

    def test_pending_request_survives_while_still_held(self):
        # No new request this frame, override still active -> pending stays queued, no new logs.
        execute_replan, pending, logs = self._call(
            replan_needed=False, collision_hold_active=True,
            pending_replan_request=True, reasons_below_threshold=True
        )
        self.assertFalse(execute_replan)
        self.assertTrue(pending)
        self.assertEqual(logs, [])

    def test_release_executes_when_reasons_still_below_threshold(self):
        # Override just cleared, and the original reason is still bad -> execute now.
        execute_replan, pending, logs = self._call(
            replan_needed=False, collision_hold_active=False,
            pending_replan_request=True, reasons_below_threshold=True
        )
        self.assertTrue(execute_replan)
        self.assertFalse(pending)
        self.assertEqual(logs, ["replan executed on release"])

    def test_release_drops_when_reasons_recovered(self):
        # Override cleared, but reasons recovered in the meantime -> drop, don't replay stale decision.
        execute_replan, pending, logs = self._call(
            replan_needed=False, collision_hold_active=False,
            pending_replan_request=True, reasons_below_threshold=False
        )
        self.assertFalse(execute_replan)
        self.assertFalse(pending)
        self.assertEqual(logs, ["replan held request dropped — reasons recovered"])

    def test_no_request_no_hold_is_noop(self):
        execute_replan, pending, logs = self._call(
            replan_needed=False, collision_hold_active=False,
            pending_replan_request=False, reasons_below_threshold=False
        )
        self.assertFalse(execute_replan)
        self.assertFalse(pending)
        self.assertEqual(logs, [])

    def test_executing_a_normal_request_never_leaves_it_pending(self):
        # Sanity check against the "double-trigger" risk called out in the spec:
        # an immediately-executed request must not also sit around as pending.
        execute_replan, pending, logs = self._call(
            replan_needed=True, collision_hold_active=False,
            pending_replan_request=False, reasons_below_threshold=True
        )
        self.assertTrue(execute_replan)
        self.assertFalse(pending)


if __name__ == '__main__':
    unittest.main()
