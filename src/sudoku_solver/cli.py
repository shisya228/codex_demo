"""Command-line interface for Sudoku solver."""

from __future__ import annotations

import argparse
from pathlib import Path

from .solver import GRID_SIZE, SudokuResult, parse_puzzle, solve_sudoku


def _format_grid(result: SudokuResult) -> str:
    lines = []
    for row in range(GRID_SIZE):
        line = " ".join(str(value) for value in result.grid[row])
        lines.append(line)
    return "\n".join(lines)


def _read_input(args: argparse.Namespace) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if args.puzzle:
        return args.puzzle
    raise SystemExit("Please provide a puzzle string or --file.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Sudoku puzzles with Numba.")
    parser.add_argument("puzzle", nargs="?", help="81-char puzzle string")
    parser.add_argument("--file", help="Path to file containing puzzle")
    args = parser.parse_args()

    raw = _read_input(args)
    puzzle = parse_puzzle(raw)
    result = solve_sudoku(puzzle)

    if not result.solved:
        raise SystemExit("No solution found.")

    print(_format_grid(result))


if __name__ == "__main__":
    main()
