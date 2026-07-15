"""Build a GenAlgSolver from a YAML/dict config instead of hand-wiring
constructor calls in every downstream script.

Only handles what's genuinely declarative: picking the solver class and
resolving importable references (assembler, fitness_function, scalarizer)
by dotted path. Anything that needs runtime computation (e.g. an alphabet
built from a project-specific database) stays the caller's job and is
passed in as an extra kwarg to build_solver().
"""

import ast
import importlib

import yaml

from navicatGA.float_solver import FloatGenAlgSolver
from navicatGA.smiles_solver import SmilesGenAlgSolver
from navicatGA.selfies_solver import SelfiesGenAlgSolver

SOLVERS = {
    "float": FloatGenAlgSolver,
    "smiles": SmilesGenAlgSolver,
    "selfies": SelfiesGenAlgSolver,
}
# XYZGenAlgSolver requires AaronTools; register it lazily so importing this
# module doesn't force that dependency on float/smiles/selfies users.
try:
    from navicatGA.xyz_solver import XYZGenAlgSolver

    SOLVERS["xyz"] = XYZGenAlgSolver
except ImportError:
    pass

# Config keys that name a dotted-path reference to resolve into a live object.
_REFERENCE_KEYS = (
    "fitness_function",
    "chromosome_to_smiles",
    "chromosome_to_selfies",
    "chromosome_to_array",
    "chromosome_to_xyz",
)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve(dotted_path: str):
    """Import 'package.module:attr' and return the attribute.

    Several of navicatGA's own example assemblers/fitness functions are
    factories rather than plain callables (e.g. concatenate_list(),
    fitness_function_selfies(1)) - append a literal '(...)' call to the
    reference to call the resolved attribute and use its return value
    instead, e.g. 'navicatGA.fitness_functions_selfies:fitness_function_selfies(1)'
    or 'navicatGA.wrappers_smiles:chromosome_to_smiles()'. Only literal
    positional/keyword arguments are supported (numbers, strings, lists,
    dicts, ... - anything ast.literal_eval accepts), not arbitrary
    expressions.
    """
    ref = dotted_path
    call_args, call_kwargs = None, None
    if ref.endswith(")") and "(" in ref:
        ref, _, call_str = ref.partition("(")
        call_node = ast.parse(f"_({call_str}", mode="eval").body
        call_args = [ast.literal_eval(a) for a in call_node.args]
        call_kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call_node.keywords}

    module_path, _, attr = ref.partition(":")
    if not attr:
        raise ValueError(f"'{dotted_path}' is not a 'module.path:attr' reference.")
    obj = getattr(importlib.import_module(module_path), attr)
    return obj(*call_args, **call_kwargs) if call_args is not None else obj


def _build_scalarizer(spec):
    if not isinstance(spec, dict):
        return spec  # already None or a live object
    cls = resolve(spec["class"])
    return cls(**spec.get("kwargs", {}))


def build_solver(solver_config: dict, **extra_params):
    """
    :param solver_config: the 'solver' block of a config (type, fitness_function,
        chromosome_to_*, scalarizer, params)
    :param extra_params: additional/overriding solver kwargs computed at runtime
        (e.g. alphabet_list), merged on top of solver_config['params']
    :return: an instantiated, unsolved GenAlgSolver subclass
    """
    solver_type = solver_config["type"]
    if solver_type not in SOLVERS:
        raise ValueError(
            f"Unknown solver type '{solver_type}'. Available: {list(SOLVERS)}"
        )

    params = {**solver_config.get("params", {}), **extra_params}
    for key in _REFERENCE_KEYS:
        if key in solver_config:
            params[key] = resolve(solver_config[key])
    if "scalarizer" in solver_config:
        params["scalarizer"] = _build_scalarizer(solver_config["scalarizer"])

    return SOLVERS[solver_type](**params)


def build_solver_from_yaml(path, **extra_params):
    config = load_yaml(path)
    return build_solver(config["solver"], **extra_params), config
