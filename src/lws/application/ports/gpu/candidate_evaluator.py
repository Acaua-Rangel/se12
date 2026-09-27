"""Port: times ONE launch configuration for one kernel/shape/batch and
checks whether its output still meets the accuracy bound — the tuner's
per-candidate measurement (US-013, AC-024: "only configurations that pass
the AC-010 accuracy check are saved").
"""

from __future__ import annotations

import typing

from lws.domain.launch.heuristic import LaunchConfiguration
from lws.domain.launch.search_space import TimingAndAccuracy


class CandidateEvaluator(typing.Protocol):
    def evaluate(self, configuration: LaunchConfiguration) -> TimingAndAccuracy:
        ...
