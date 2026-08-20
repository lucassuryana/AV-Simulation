import unittest

from scenarios.overtaking_cyclist_bidirectional_road import filter_safe_trajectories


class SafetyFilterTestCase(unittest.TestCase):

    def _call(self, trajectories, unsafe_set):
        logs = []
        result = filter_safe_trajectories(
            trajectories, is_unsafe_fn=lambda i, t: t in unsafe_set, log_fn=logs.append
        )
        return result, logs

    def test_all_safe_passes_through_unchanged(self):
        trajectories = ["a", "b", "c"]
        result, logs = self._call(trajectories, unsafe_set=set())
        self.assertEqual(result, ["a", "b", "c"])
        self.assertEqual(logs, [])

    def test_excludes_only_the_unsafe_candidates(self):
        trajectories = ["a", "b", "c", "d"]
        result, logs = self._call(trajectories, unsafe_set={"b", "d"})
        self.assertEqual(result, ["a", "c"])
        self.assertEqual(logs, ["Safety filter: excluded 2 of 4 candidates"])

    def test_preserves_original_order_of_survivors(self):
        # Guards against an accidental reorder that would desync scores/indices
        # downstream (step 5's requirement that indices stay consistent).
        trajectories = [0, 1, 2, 3, 4]
        result, _ = self._call(trajectories, unsafe_set={1, 3})
        self.assertEqual(result, [0, 2, 4])

    def test_falls_back_to_last_candidate_when_everything_unsafe(self):
        # perform_replan always appends the implicit follow_trajectory last;
        # it must survive as the fallback even though the filter marked it unsafe too.
        trajectories = ["overtake_a", "overtake_b", "follow_trajectory"]
        result, logs = self._call(trajectories, unsafe_set=set(trajectories))
        self.assertEqual(result, ["follow_trajectory"])
        self.assertEqual(logs, [
            "Safety filter: excluded 3 of 3 candidates",
            "Safety filter excluded all candidates — falling back to follow_trajectory",
        ])

    def test_empty_input_is_a_noop(self):
        result, logs = self._call([], unsafe_set=set())
        self.assertEqual(result, [])
        self.assertEqual(logs, [])

    def test_single_survivor_no_exclusion_log_when_nothing_excluded(self):
        trajectories = ["a"]
        result, logs = self._call(trajectories, unsafe_set=set())
        self.assertEqual(result, ["a"])
        self.assertEqual(logs, [])

    def test_is_unsafe_fn_receives_index_into_original_list(self):
        # Callers need the index to special-case the last candidate (follow_trajectory)
        # the same way the scoring code resamples it differently from real maneuvers.
        trajectories = ["overtake_a", "overtake_b", "follow_trajectory"]
        seen_indices = []

        def is_unsafe(i, t):
            seen_indices.append(i)
            return False

        result = filter_safe_trajectories(trajectories, is_unsafe, log_fn=None)
        self.assertEqual(seen_indices, [0, 1, 2])
        self.assertEqual(result, trajectories)


if __name__ == '__main__':
    unittest.main()
