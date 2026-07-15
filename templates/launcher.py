#!/usr/bin/env python3
"""Generic launcher template: build a solver from a YAML config and run it.

Copy this into your project alongside your own config.yaml (start from
templates/{float,smiles,selfies,xyz}.yaml) and assembler/fitness module
(start from templates/assembler_and_fitness.py), then adjust to taste - the
things ga_flp/launcher.py adds on top (per-cycle logging, output files,
loading an external model) are project-specific and belong in your copy,
not here.
"""
import argparse

from navicatGA.config import build_solver_from_yaml


def main():
    parser = argparse.ArgumentParser(
        description="Run a navicatGA optimization from a YAML config."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument(
        "--num_cycles",
        type=int,
        default=1,
        help="Number of solve() calls; >1 runs one generation at a time so "
        "you can inspect/log progress between cycles (see ga_flp/launcher.py).",
    )
    args = parser.parse_args()

    # If your alphabet/starting population is computed at runtime (e.g. read
    # from a database - the common case for smiles/xyz), build it here and
    # pass it in; it overrides/extends the YAML's params block:
    #
    # alphabet_list = my_alphabet_builder(...)
    # solver, config = build_solver_from_yaml(args.config, alphabet_list=alphabet_list)
    solver, config = build_solver_from_yaml(args.config)

    if args.num_cycles > 1:
        for cycle in range(args.num_cycles):
            solver.solve(1)
            print(f"[cycle {cycle}] best fitness so far: {solver.best_fitness_}")
    else:
        solver.solve()

    print(f"Best individual: {solver.best_individual_}")
    print(f"Best fitness: {solver.best_fitness_}")
    solver.close_solver_logger()


if __name__ == "__main__":
    main()
