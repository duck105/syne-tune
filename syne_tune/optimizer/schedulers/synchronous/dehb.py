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
import logging
import numpy as np
from dataclasses import dataclass

from syne_tune.optimizer.schedulers.synchronous.dehb_bracket_manager import (
    DifferentialEvolutionHyperbandBracketManager,
)
from syne_tune.optimizer.schedulers.synchronous.hyperband_bracket import (
    SlotInRung,
)
from syne_tune.optimizer.scheduler import TrialSuggestion, SchedulerDecision
from syne_tune.optimizer.schedulers.fifo import ResourceLevelsScheduler
from syne_tune.backend.trial_status import Trial
from syne_tune.config_space import cast_config_values
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    check_and_merge_defaults,
    Categorical,
    String,
    assert_no_invalid_options,
    Integer,
    Float,
)
from syne_tune.optimizer.schedulers.random_seeds import RandomSeedGenerator
from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher
from syne_tune.optimizer.schedulers.searchers.searcher_factory import searcher_factory
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory import (
    make_hyperparameter_ranges,
)

__all__ = ["DifferentialEvolutionHyperbandScheduler"]

logger = logging.getLogger(__name__)


_ARGUMENT_KEYS = {
    "searcher",
    "search_options",
    "metric",
    "mode",
    "points_to_evaluate",
    "random_seed",
    "max_resource_attr",
    "resource_attr",
    "mutation_factor",
    "crossover_probability",
}

_DEFAULT_OPTIONS = {
    "searcher": "random",
    "mode": "min",
    "resource_attr": "epoch",
    "searcher_data": "rungs",
    "mutation_factor": 0.5,
    "crossover_probability": 0.5,
}

_CONSTRAINTS = {
    "metric": String(),
    "mode": Categorical(choices=("min", "max")),
    "random_seed": Integer(0, 2**32 - 1),
    "max_resource_attr": String(),
    "resource_attr": String(),
    "mutation_factor": Float(lower=0, upper=1),
    "crossover_probability": Float(lower=0, upper=1),
}


@dataclass
class TrialInformation:
    """
    Information the scheduler maintains per trial.
    """

    encoded_config: np.ndarray
    level: int
    metric_val: Optional[float] = None


class ExtendedSlotInRung:
    """
    Extends :class:`SlotInRung` mostly for convenience
    """

    def __init__(self, bracket_id: int, slot_in_rung: SlotInRung):
        self.bracket_id = bracket_id
        self.rung_index = slot_in_rung.rung_index
        self.level = slot_in_rung.level
        self.slot_index = slot_in_rung.slot_index
        self.trial_id = slot_in_rung.trial_id
        self.metric_val = slot_in_rung.metric_val
        self.do_selection = False

    def slot_in_rung(self) -> SlotInRung:
        return SlotInRung(
            rung_index=self.rung_index,
            level=self.level,
            slot_index=self.slot_index,
            trial_id=self.trial_id,
            metric_val=self.metric_val,
        )


class DifferentialEvolutionHyperbandScheduler(ResourceLevelsScheduler):
    """
    Differential Evolution Hyperband, as proposed in

        DEHB: Evolutionary Hyperband for Scalable, Robust and Efficient
        Hyperparameter Optimization
        Noor Awad, Neeratyoy Mallik, Frank Hutter
        IJCAI 30 (2021)
        Pages 2147-2153
        https://arxiv.org/abs/2105.09821

    We implement DEHB as a variant of synchronous Hyperband, which may
    differ slightly from the implementation of the authors.

    Main differences to synchronous Hyperband:

    * In DEHB, trials are not paused and potentially promoted (except in the
        very first bracket, which can be ignored). Therefore, checkpointing
        (even if implemented for the objective) is not used
    * Only the initial configurations are drawn at random (or drawn from the
        the searcher). Whenever possible, new configurations (in their
        internal encoding) are derived from earlier ones by way of
        differential evolution

    Parameters
    ----------
    config_space : dict
        Configuration space for trial evaluation function
    rungs_first_bracket : List[Tuple[int, int]]
        Determines rung level systems for each bracket, see
        :class:`DifferentialEvolutionHyperbandBracketManager`
    num_brackets_per_iteration : Optional[int]
        Number of brackets per iteration. The algorithm cycles through
        these brackets in one iteration. If not given, the maximum
        number is used (i.e., `len(rungs_first_bracket)`)
    searcher : str
        Selects searcher. Passed to `searcher_factory`.
        NOTE: Different to :class:`FIFOScheduler`, we do not accept a
        `BaseSearcher` object here.
    search_options : dict
        Passed to `searcher_factory`
    metric : str
        Name of metric to optimize, key in result's obtained via
        `on_trial_result`
    mode : str
        Mode to use for the metric given, can be 'min' or 'max'
    points_to_evaluate: list[dict] or None
        See :class:`SynchronousHyperbandScheduler`.
        Note that this list is only used for initial configurations. Once
        DEHB starts to combine new configs from earlier ones, the list here
        is ignored, even if it still contains entries.
    random_seed : int
        Master random seed. Generators used in the scheduler or searcher are
        seeded using `RandomSeedGenerator`. If not given, the master random
        seed is drawn at random here.
    max_resource_attr : str
        Key name in config for fixed attribute containing the maximum resource.
        If given, trials need not be stopped, which can run more efficiently.
    resource_attr : str
        Name of resource attribute in result's obtained via `on_trial_result`.
        Note: The type of resource must be int.
    mutation_factor : float, in (0, 1]
        Factor F used in the rand/1 mutation operation of DE
    crossover_probability : float, in (0, 1)
        Probability p used in crossover operation (child entries are chosen
        with probability p)
    """

    def __init__(
        self,
        config_space: dict,
        rungs_first_bracket: List[Tuple[int, int]],
        num_brackets_per_iteration: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(config_space)
        self._create_internal(rungs_first_bracket, num_brackets_per_iteration, **kwargs)

    def _create_internal(
        self,
        rungs_first_bracket: List[Tuple[int, int]],
        num_brackets_per_iteration: Optional[int] = None,
        **kwargs,
    ):
        # Check values and impute default values
        assert_no_invalid_options(
            kwargs, _ARGUMENT_KEYS, name="DifferentialEvolutionHyperbandScheduler"
        )
        kwargs = check_and_merge_defaults(
            kwargs,
            set(),
            _DEFAULT_OPTIONS,
            _CONSTRAINTS,
            dict_name="scheduler_options",
        )
        self.metric = kwargs.get("metric")
        assert self.metric is not None, (
            "Argument 'metric' is mandatory. Pass the name of the metric "
            + "reported by your training script, which you'd like to "
            + "optimize, and use 'mode' to specify whether it should "
            + "be minimized or maximized"
        )
        self.mode = kwargs["mode"]
        self.max_resource_attr = kwargs.get("max_resource_attr")
        self._resource_attr = kwargs["resource_attr"]
        self.mutation_factor = kwargs["mutation_factor"]
        self.crossover_probability = kwargs["crossover_probability"]
        # Generator for random seeds
        random_seed = kwargs.get("random_seed")
        if random_seed is None:
            random_seed = np.random.randint(0, 2**32)
        logger.info(f"Master random_seed = {random_seed}")
        self.random_seed_generator = RandomSeedGenerator(random_seed)
        # Generate searcher
        searcher = kwargs["searcher"]
        assert isinstance(
            searcher, str
        ), f"searcher must be of type string, but has type {type(searcher)}"
        search_options = kwargs.get("search_options")
        if search_options is None:
            search_options = dict()
        else:
            search_options = search_options.copy()
        search_options.update(
            {
                "config_space": self.config_space.copy(),
                "metric": self.metric,
                "points_to_evaluate": kwargs.get("points_to_evaluate"),
                "scheduler_mode": kwargs["mode"],
                "random_seed_generator": self.random_seed_generator,
                "resource_attr": self._resource_attr,
                "scheduler": "hyperband_synchronous",
            }
        )
        if searcher == "bayesopt":
            # We need `max_epochs` in this case
            max_epochs = self._infer_max_resource_level(
                max_resource_level=None, max_resource_attr=self.max_resource_attr
            )
            assert max_epochs is not None, (
                "If searcher='bayesopt', need to know the maximum resource "
                + "level. Please provide max_resource_attr argument."
            )
            search_options["max_epochs"] = max_epochs
        self.searcher: BaseSearcher = searcher_factory(searcher, **search_options)
        # Bracket manager
        self.bracket_manager = DifferentialEvolutionHyperbandBracketManager(
            rungs_first_bracket=rungs_first_bracket,
            mode=self.mode,
            num_brackets_per_iteration=num_brackets_per_iteration,
        )
        # Needed to convert encoded configs to configs
        self._hp_ranges = make_hyperparameter_ranges(self.config_space)
        # PRNG for mutation and crossover random draws
        self.random_state = np.random.RandomState(self.random_seed_generator())
        # How often is selection skipped because target still pending?
        self.num_selection_skipped = 0
        # Maps trial_id to ext_slot, as returned by `bracket_manager.next_job`,
        # and required by `bracket_manager.on_result`. Entries are removed once
        # passed to `on_result`. Here, ext_slot is of type `ExtendedSlotInRung`.
        self._trial_to_pending_slot = dict()
        # Maps trial_id to trial information (in particular, the encoded
        # config)
        self._trial_info = dict()
        # Maps level to list of trial_ids of all completed jobs (so that
        # metric values are available). This global "parent pool" is used
        # during mutations if the normal parent pool is too small
        self._global_parent_pool = {level: [] for _, level in rungs_first_bracket}

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        do_debug_log = self.searcher.debug_log is not None
        if do_debug_log and trial_id == 0:
            # This is printed at the start of the experiment. Cannot do this
            # at construction, because with `RemoteLauncher` this does not end
            # up in the right log
            parts = ["Rung systems for each bracket:"] + [
                f"Bracket {bracket}: {rungs}"
                for bracket, rungs in enumerate(self.bracket_manager.bracket_rungs)
            ]
            logger.info("\n".join(parts))
        # Ask bracket manager for job
        bracket_id, slot_in_rung = self.bracket_manager.next_job()
        ext_slot = ExtendedSlotInRung(bracket_id, slot_in_rung)
        is_base_rung = ext_slot.rung_index == 0  # Slot in base rung?
        if ext_slot.trial_id == trial_id and is_base_rung:
            # At the very start, we draw configs from the searcher
            encoded_config = self._encoded_config_from_searcher(trial_id)
        elif bracket_id == 0:
            # First bracket, but not base rung. Promotion as in synchronous
            # HB, but we assign new trial_id
            encoded_config = self._encoded_config_by_promotion(ext_slot)
        else:
            # Here, we can do DE (mutation, crossover)
            encoded_config = self._extended_config_by_mutation_crossover(
                trial_id, ext_slot
            )
        if encoded_config is not None:
            suggestion = self._register_new_config_as_pending(
                trial_id=trial_id, ext_slot=ext_slot, encoded_config=encoded_config
            )
            if do_debug_log:
                logger.info(
                    f"trial_id {trial_id} starts (milestone = " f"{ext_slot.level})"
                )
        else:
            # Searcher failed to return a config for a new trial_id. We report
            # the corresponding job as failed, so that in case the experiment
            # is continued, the bracket is not blocked with a slot which remains
            # pending forever
            logger.warning(
                "Searcher failed to suggest a configuration for new trial "
                f"{trial_id}. The corresponding slot is marked as failed."
            )
            self._report_as_failed(ext_slot)
            suggestion = None
        return suggestion

    def _encoded_config_from_searcher(self, trial_id: int) -> np.ndarray:
        config = self.searcher.get_config(trial_id=str(trial_id))
        if config is not None:
            encoded_config = self._hp_ranges.to_ndarray(config)
        else:
            encoded_config = None
        return encoded_config

    def _encoded_config_by_promotion(self, ext_slot: ExtendedSlotInRung) -> np.ndarray:
        parent_trial_id = self.bracket_manager.top_of_previous_rung(
            bracket_id=ext_slot.bracket_id, pos=ext_slot.slot_index
        )
        return self._trial_info[parent_trial_id].encoded_config

    def _extended_config_by_mutation_crossover(
        self, trial_id: int, ext_slot: ExtendedSlotInRung
    ) -> np.ndarray:
        assert trial_id > ext_slot.trial_id  # Sanity check
        ext_slot.do_selection = True
        target_trial_id = self.bracket_manager.trial_id_from_parent_slot(
            bracket_id=ext_slot.bracket_id,
            level=ext_slot.level,
            slot_index=ext_slot.slot_index,
        )
        mutant = self._mutation(ext_slot)
        return self._crossover(
            mutant=mutant,
            target=self._trial_info[target_trial_id].encoded_config,
        )

    def _register_new_config_as_pending(
        self, trial_id: int, ext_slot: ExtendedSlotInRung, encoded_config: np.ndarray
    ) -> TrialSuggestion:
        # Register as pending
        self._trial_to_pending_slot[trial_id] = ext_slot
        # Register new trial_id
        self._trial_info[trial_id] = TrialInformation(
            encoded_config=encoded_config,
            level=ext_slot.level,
        )
        # Return new config
        config = cast_config_values(
            self._hp_ranges.from_ndarray(encoded_config), self.config_space
        )
        if self.max_resource_attr is not None:
            config = dict(config, **{self.max_resource_attr: ext_slot.level})
        return TrialSuggestion.start_suggestion(config=config)

    def _report_as_failed(self, ext_slot: ExtendedSlotInRung):
        result_failed = ext_slot.slot_in_rung()
        result_failed.metric_val = np.NAN
        self.bracket_manager.on_result((ext_slot.bracket_id, result_failed))

    def _on_trial_result(
        self, trial: Trial, result: dict, call_searcher: bool = True
    ) -> str:
        trial_id = trial.trial_id
        if trial_id in self._trial_to_pending_slot:
            ext_slot = self._trial_to_pending_slot[trial_id]
            bracket_id = ext_slot.bracket_id
            milestone = ext_slot.level
            metric_val, resource = self._extract_from_result(trial_id, result)
            trial_decision = SchedulerDecision.CONTINUE
            if resource >= milestone:
                assert resource == milestone, (
                    f"Trial trial_id {trial_id}: Obtained result for "
                    + f"resource = {resource}, but not for {milestone}. "
                    + "Training script must not skip rung levels!"
                )
                self._record_new_metric_value(trial_id, milestone, metric_val)
                # Selection
                winner_trial_id = self._selection(trial_id, ext_slot, metric_val)
                # Return updated slot information to bracket
                ext_slot.trial_id = winner_trial_id
                ext_slot.metric_val = self._trial_info[winner_trial_id].metric_val
                slot_in_rung = ext_slot.slot_in_rung()
                self.bracket_manager.on_result((bracket_id, slot_in_rung))
                trial_decision = SchedulerDecision.STOP
                if call_searcher:
                    config = self._hp_ranges.from_ndarray(
                        self._trial_info[trial_id].encoded_config
                    )
                    self.searcher.on_trial_result(
                        trial_id=str(trial_id),
                        config=config,
                        result=result,
                        update=True,
                    )
        else:
            trial_decision = SchedulerDecision.STOP
            logger.warning(
                f"Received result for trial_id {trial_id}, which is not "
                f"pending. This result is not used:\n{result}"
            )

        return trial_decision

    def _extract_from_result(self, trial_id: int, result: dict) -> (float, int):
        assert self.metric in result, (
            f"Result for trial_id {trial_id} does not contain "
            + f"'{self.metric}' field"
        )
        metric_val = float(result[self.metric])
        assert self._resource_attr in result, (
            f"Result for trial_id {trial_id} does not contain "
            + f"'{self._resource_attr}' field"
        )
        resource = int(result[self._resource_attr])
        return metric_val, resource

    def _record_new_metric_value(
        self, trial_id: int, milestone: int, metric_val: float
    ):
        # Record metric value
        trial_info = self._trial_info[trial_id]
        assert trial_info.level == milestone  # Sanity check
        trial_info.metric_val = metric_val
        # Update global parent pool
        self._global_parent_pool[milestone].append(trial_id)

    def _mutation(self, ext_slot: ExtendedSlotInRung) -> np.ndarray:
        bracket_id = ext_slot.bracket_id
        level = ext_slot.level
        assert bracket_id > 0
        orig_pool_size = self.bracket_manager.size_of_current_rung(bracket_id)
        pool_size = orig_pool_size
        global_pool = None
        if orig_pool_size < 3:
            global_pool = self._global_parent_pool[level]
            pool_size += len(global_pool)
            # If this is still too small, we add parent pools of other levels
            for _, other_level in self.bracket_manager.bracket_rungs[0]:
                if pool_size >= 3:
                    break
                if other_level != level:
                    extra_pool = self._global_parent_pool[other_level]
                    global_pool = global_pool + extra_pool
                    pool_size += len(extra_pool)
            # TODO: If this happens, have to do something else here. For example,
            # could pick trial_id's which are still pending
            assert pool_size >= 3, f"Cannot compose parent pool of size >= 3"
        # Sample entries from parent pool
        positions = list(self.random_state.choice(pool_size, 3, replace=False))
        is_base_rung = ext_slot.rung_index == 0
        parent_trial_ids = []
        for pos in positions:
            if pos >= orig_pool_size:
                trial_id = global_pool[pos - orig_pool_size]
            elif is_base_rung:
                trial_id = self.bracket_manager.trial_id_from_parent_slot(
                    bracket_id=bracket_id, level=level, slot_index=pos
                )
            else:
                trial_id = self.bracket_manager.top_of_previous_rung(
                    bracket_id=bracket_id, pos=pos
                )
            parent_trial_ids.append(trial_id)
        return self._de_mutation(parent_trial_ids)

    def _de_mutation(self, parent_trial_ids: List[int]) -> np.ndarray:
        ec1 = self._trial_info[parent_trial_ids[0]].encoded_config
        ec2 = self._trial_info[parent_trial_ids[1]].encoded_config
        ec3 = self._trial_info[parent_trial_ids[2]].encoded_config
        mutant = (ec2 - ec3) * self.mutation_factor + ec1
        # Entries which violate boundaries are resampled at random
        violations = np.where((mutant > 1) | (mutant < 0))[0]
        if len(violations) > 0:
            mutant[violations] = self.random_state.uniform(
                low=0.0, high=1.0, size=len(violations)
            )
        return mutant

    def _crossover(self, mutant: np.ndarray, target: np.ndarray) -> np.ndarray:
        dimensions = self._hp_ranges.ndarray_size
        cross_points = self.random_state.rand(dimensions) < self.crossover_probability
        if not np.any(cross_points):
            cross_points[self.random_state.randint(0, dimensions)] = True
        # For any HP whose encoding has dimension > 1 (e.g., categorical), we
        # make sure not to cross-over inside the encoding
        for (start, end) in self._hp_ranges.encoded_ranges.values():
            if end > start + 1:
                cross_points[(start + 1) : end] = cross_points[start]
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def _selection(
        self, trial_id: int, ext_slot: ExtendedSlotInRung, metric_val: float
    ) -> int:
        winner_trial_id = trial_id
        if ext_slot.do_selection:
            bracket_id = ext_slot.bracket_id
            # Note: This can have changed since crossover
            target_trial_id = self.bracket_manager.trial_id_from_parent_slot(
                bracket_id=bracket_id,
                level=ext_slot.level,
                slot_index=ext_slot.slot_index,
            )
            target_metric_val = self._trial_info[target_trial_id].metric_val
            if target_metric_val is not None:
                # Selection
                metric_sign = -1 if self.mode == "max" else 1
                if metric_sign * (metric_val - target_metric_val) >= 0:
                    winner_trial_id = target_trial_id
            else:
                # target has no metric value yet. This should not happen often
                logger.warning(
                    "Could not do selection because target metric "
                    f"value still pending (bracket_id = {bracket_id}"
                    f", trial_id = {trial_id}, "
                    f"target_trial_id = {target_trial_id}"
                )
                self.num_selection_skipped += 1
        return winner_trial_id

    def on_trial_result(self, trial: Trial, result: dict) -> str:
        return self._on_trial_result(trial, result, call_searcher=True)

    def on_trial_error(self, trial: Trial):
        """
        Given the `trial` is currently pending, we send a result at its
        milestone for metric value NaN. Such trials are ranked after all others
        and will most likely not be promoted.

        """
        trial_id = trial.trial_id
        self.searcher.evaluation_failed(str(trial_id))
        if trial_id in self._trial_to_pending_slot:
            ext_slot = self._trial_to_pending_slot[trial_id]
            self._report_as_failed(ext_slot)
        else:
            logger.warning(
                f"Trial trial_id {trial_id} not registered at pending: "
                "on_trial_error call is ignored"
            )

    def metric_names(self) -> List[str]:
        return [self.metric]

    def metric_mode(self) -> str:
        return self.mode
