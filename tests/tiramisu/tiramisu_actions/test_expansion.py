import tests.utils as test_utils
from tiralib.tiramisu.schedule import Schedule
from tiralib.tiramisu.tiramisu_actions import Expansion, Parallelization
from tiralib.config import BaseConfig


def test_expansion_init():
    expansion = Expansion(["comp"])
    assert expansion.computation == "comp"


def test_initialize_action_for_tree():
    BaseConfig.init()
    sample = test_utils.expansion_sample()
    expansion = Expansion(["addition"])
    expansion.initialize_action_for_tree(sample.tree)
    assert expansion.computation == "addition"
    assert expansion.tree, "Tree should not be None"


def test_set_string_representation():
    BaseConfig.init()
    sample = test_utils.expansion_sample()
    expansion = Expansion(["addition"])
    expansion.initialize_action_for_tree(sample.tree)
    assert expansion.tiramisu_optim_str == "addition.expand(true);\n"


def test_get_candidates():
    BaseConfig.init()
    sample = test_utils.expansion_sample()
    schedule = Schedule(sample)
    candidates = Expansion.get_computation_candidates(schedule)
    assert candidates == ["addition"]


def test_expansion_application():
    BaseConfig.init()
    sample = test_utils.expansion_sample()
    schedule = Schedule(sample)
    schedule.add_optimizations([Parallelization([("addition", 1)])])

    assert not schedule.is_legal()

    schedule = Schedule(sample)
    schedule.add_optimizations(
        [Expansion(["addition"]), Parallelization([("addition", 1)])]
    )
    assert schedule.is_legal()


def test_expansion_along_loop_string_representation():
    BaseConfig.init()
    sample = test_utils.expansion_sample()
    expansion = Expansion([("addition", 1)])
    expansion.initialize_action_for_tree(sample.tree)
    assert expansion.computation == "addition"
    assert str(expansion) == "E(L1,comps=['addition'])"
    assert "expand_buffer_along_loop(&addition, 1)" in expansion.tiramisu_optim_str


def test_expansion_from_sched_str():
    BaseConfig.init()
    sample = test_utils.expansion_sample()
    for sched_str in ["E(L1,comps=['addition'])", "E(comps=['addition'])"]:
        schedule = Schedule.from_sched_str(sched_str, sample)
        assert str(schedule) == sched_str


def _server_program(name):
    from pathlib import Path
    from tiralib.tiramisu.tiramisu_program import TiramisuProgram

    return TiramisuProgram.init_server(
        cpp_code=Path(f"examples/{name}_generator.cpp").read_text(),
        load_isl_ast=True,
        load_tree=True,
        reuse_server=True,
    )


def test_expansion_along_loop_enables_parallelization():
    # symm accumulates into a scalar temp at every (i, j): j cannot be
    # parallelized or interchanged until temp is expanded along j.
    BaseConfig.init()
    sample = _server_program("function_symm_MINI")
    comps = "comps=['temp_init','C_r','temp_comp','C_out']"
    assert not Schedule.from_sched_str(f"P(L1,{comps})", sample).is_legal()
    assert not Schedule.from_sched_str(f"I(L0,L1,{comps})", sample).is_legal()
    expand = "E(L1,comps=['temp_init'])"
    assert Schedule.from_sched_str(expand, sample).is_legal()
    assert Schedule.from_sched_str(f"{expand}|P(L1,{comps})", sample).is_legal()
    assert Schedule.from_sched_str(f"{expand}|I(L0,L1,{comps})", sample).is_legal()
    # Expanding the same buffer again along the same loop changes nothing.
    again = f"{expand}|E(L1,comps=['temp_comp'])"
    assert Schedule.from_sched_str(f"{again}|P(L1,{comps})", sample).is_legal()
    # Each j iteration now has its own copy, but successive i still share it.
    assert not Schedule.from_sched_str(f"{expand}|P(L0,{comps})", sample).is_legal()


def test_expansion_along_loop_refused():
    BaseConfig.init()
    sample = _server_program("function_durbin_MINI")
    # sum and z only live within one iteration of k: they can be expanded.
    assert Schedule.from_sched_str("E(L0,comps=['sum_1'])", sample).is_legal()
    assert Schedule.from_sched_str("E(L0,comps=['z'])", sample).is_legal()
    # alpha is read by the next iteration of k.
    assert not Schedule.from_sched_str("E(L0,comps=['alpha_1'])", sample).is_legal()
    # z's reader y_1 is not in z's i loop.
    assert not Schedule.from_sched_str("E(L1,comps=['z'])", sample).is_legal()
