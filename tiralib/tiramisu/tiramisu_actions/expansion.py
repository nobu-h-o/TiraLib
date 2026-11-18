from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, List

from tiralib.tiramisu.compiling_service import CompilingService
from tiralib.tiramisu.tiramisu_iterator_node import IteratorIdentifier
from tiralib.tiramisu.tiramisu_tree import TiramisuTree
from tiralib.config import BaseConfig

if TYPE_CHECKING:
    from tiralib.tiramisu.schedule import Schedule

from tiralib.tiramisu.tiramisu_actions.tiramisu_action import (
    TiramisuAction,
    TiramisuActionType,
)


class Expansion(TiramisuAction):
    """
    Expansion optimization command.

    Two forms:

    - ``Expansion(["comp"])``: Tiramisu's ``expand()`` of that one
      computation (``E(comps=['comp'])``).
    - ``Expansion([("comp", level)])``: expand the temporary buffer written by
      ``comp`` along the loop at ``level`` enclosing it
      (``E(L<level>,comps=['comp'])``). Every computation writing that buffer,
      and every read of it, gets the loop iterator as a new outermost index,
      so each iteration of the loop works on its own copy; a scalar becomes a
      1D array. The compiler refuses it (illegal) unless every read gets its
      value from the same iteration of the loop. It works on the original
      loops, so it must come before every other transformation.
    """

    def __init__(self, params: List[str] | List[IteratorIdentifier]):
        assert len(params) == 1
        self.iterator_id: IteratorIdentifier | None = None
        if isinstance(params[0], tuple):
            self.iterator_id = params[0]
            self.computation = params[0][0]
        else:
            assert isinstance(params[0], str)
            self.computation = params[0]

        super().__init__(
            type=TiramisuActionType.EXPANSION, params=params, comps=[self.computation]
        )

    def initialize_action_for_tree(self, tiramisu_tree: TiramisuTree):
        # clone the tree to be able to restore it later
        self.tree = copy.deepcopy(tiramisu_tree)
        if self.iterator_id is not None and self.iterator_id not in tiramisu_tree.iterators:
            self.iterator_id = tiramisu_tree.get_iterator_of_computation(
                *self.iterator_id
            ).id

        self.set_string_representations(tiramisu_tree)

    def set_string_representations(self, tiramisu_tree: TiramisuTree):
        assert self.computation is not None

        if self.iterator_id is None:
            self.tiramisu_optim_str = f"{self.computation}.expand(true);\n"
            self.str_representation = f"E(comps={[self.computation]})"
        else:
            level = self.iterator_id[1]
            # The server (TiraLibCPP) maps the tree level to Tiramisu's level,
            # skipping loops that run once; these snippets use it as is.
            expand_call = f"global::get_implicit_function()->expand_buffer_along_loop(&{self.computation}, {level})"  # noqa: E501
            prepare = "prepare_schedules_for_legality_checks(true);\n    perform_full_dependency_analysis();\n"  # noqa: E501
            self.tiramisu_optim_str = f"{prepare}    {expand_call};\n"
            self.legality_check_string = f"{prepare}    is_legal &= {expand_call};\n"
            self.str_representation = f"E(L{level},comps={[self.computation]})"
            return

        self.legality_check_string = self.tiramisu_optim_str

    @classmethod
    def get_computation_candidates(cls, schedule: Schedule) -> List[str]:
        assert schedule.tree, "Tree is not initialized"
        assert BaseConfig.base_config, "BaseConfig is not initialized"

        candidates: list[str] = []
        candidates_code = ""

        for optim in schedule.optims_list:
            candidates_code += "    " + optim.tiramisu_optim_str

        for comp in schedule.tree.computations:
            candidates_code += (
                f'    std::cout << "{comp}|" << {comp}.expandable() << std::endl;\n'  # noqa
            )

        cpp_code = schedule.tiramisu_program.cpp_code.replace(
            schedule.tiramisu_program.code_gen_line, candidates_code
        )

        output_path = os.path.join(
            BaseConfig.base_config.workspace,
            f"{schedule.tiramisu_program.name}_expansion_candidates",
        )

        candidates_results_str = CompilingService.run_cpp_code(
            cpp_code=cpp_code, output_path=output_path
        )
        for str_line in candidates_results_str.split("\n"):
            if str_line:
                computation_name, is_expandable = str_line.split("|")
                if is_expandable == "1":
                    candidates.append(computation_name)
        return candidates
