# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from typing import Optional, List, Tuple

from syne_tune.optimizer.schedulers.synchronous.hyperband_bracket import (
    SynchronousBracket,
    get_top_list,
)


class DifferentialEvolutionHyperbandBracket(SynchronousBracket):
    """
    Represents a bracket in Differential Evolution Hyperband (DEHB).

    There are a number of differences to brackets in standard synchronous
    Hyperband (:class:`SynchronousHyperbandBracket`):

    * The trial_ids of all rungs are initialized at construction. Either,
        they are provided, or they are initialized as 0, 1, 2, ... from
        bottom up. The trial_id fields are never None.
    * `on_result`: `result.trial_id` overwrites `trial_id` in rung even if
        latter is not None.
    * Promotions are not triggered automatically when a rung is complete
    * Some additional methods
    """

    def __init__(
        self,
        rungs: List[Tuple[int, int]],
        mode: str,
        trial_ids_for_rungs: Optional[List[List[int]]] = None,
    ):
        self.assert_check_rungs(rungs)
        super().__init__(mode)
        # Represents rung levels by (rung, level), where rung is a list of
        # (trial_id, metric_val) tuples for all rungs. The trial_id values
        # are set up front, but are in general overwritten later on
        if trial_ids_for_rungs is None:
            trial_ids_for_rungs = []
            next_trial_id = 0
            for size, _ in rungs:
                trial_ids_for_rungs.append(
                    list(range(next_trial_id, next_trial_id + size))
                )
                next_trial_id += size
        else:
            assert len(rungs) == len(trial_ids_for_rungs), (rungs, trial_ids_for_rungs)
            for i, (trial_ids, (size, _)) in enumerate(zip(trial_ids_for_rungs, rungs)):
                assert (
                    len(trial_ids) == size
                ), f"{i}: len(trial_ids) = {len(trial_ids)} != {size}"
        self._rungs = [
            ([(trial_id, None) for trial_id in trial_ids], level)
            for trial_ids, (_, level) in zip(trial_ids_for_rungs, rungs)
        ]

    @property
    def num_rungs(self) -> int:
        return len(self._rungs)

    def _current_rung_and_level(
        self,
    ) -> (List[Tuple[Optional[int], Optional[float]]], int):
        return self._rungs[self.current_rung]

    def size_of_current_rung(self) -> int:
        return len(self._current_rung_and_level()[0])

    def trial_id_for_slot(self, rung_index: int, slot_index: int) -> int:
        rung, _ = self._rungs[rung_index]
        return rung[slot_index][0]

    def top_list_for_previous_rung(self) -> List[int]:
        """
        Returns list of trial_ids correspondind to best scoring entries
        in rung below the currently active one (which must not be the base
        rung). The list is of the size of the current rung.
        """
        assert self.current_rung > 0, "Current rung is base rung"
        rung, _ = self._rungs[self.current_rung - 1]
        new_len = len(self._current_rung_and_level()[0])
        return get_top_list(rung, new_len)

    def _promote_trials_at_rung_complete(self):
        pass
