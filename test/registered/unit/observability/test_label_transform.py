"""Unit tests for label_transform — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=4, suite="stage-b-test-cpu-intel")

import unittest

from sglang.srt.observability.label_transform import (
    UNKNOWN_PRIORITY_VALUE,
    transform_priority,
)
from sglang.test.test_utils import CustomTestCase


class TestTransformPriority(CustomTestCase):
    """Test cases for transform_priority."""

    def test_label_vocabulary(self):
        """The metric label vocabulary is bounded: None, out-of-range, and
        in-range priorities map to their pinned labels."""
        self.assertEqual(transform_priority(None), UNKNOWN_PRIORITY_VALUE)
        for priority, expected in (
            (-1, "P0_Critical"),
            (0, "P0_Critical"),
            (9, "P0_Critical"),
            (10, "P1_High"),
            (29, "P1_High"),
            (30, "P2_Normal"),
            (49, "P2_Normal"),
            (50, "P3_Batch"),
            (1000000, "P3_Batch"),
        ):
            with self.subTest(priority=priority):
                self.assertEqual(transform_priority(priority), expected)


if __name__ == "__main__":
    unittest.main()
