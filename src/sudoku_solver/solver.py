"""Numba-accelerated Sudoku solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numba import njit

GRID_SIZE = 9
SUBGRID_SIZE = 3


@njit(cache=True)
def _find_empty(grid: np.ndarray) -> tuple[int, int]:
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if grid[row, col] == 0:
                return row, col
    return -1, -1


@njit(cache=True)
def _is_valid(grid: np.ndarray, row: int, col: int, value: int) -> bool:
    for i in range(GRID_SIZE):
        if grid[row, i] == value:
            return False
        if grid[i, col] == value:
            return False
    start_row = (row // SUBGRID_SIZE) * SUBGRID_SIZE
    start_col = (col // SUBGRID_SIZE) * SUBGRID_SIZE
    for r in range(start_row, start_row + SUBGRID_SIZE):
        for c in range(start_col, start_col + SUBGRID_SIZE):
            if grid[r, c] == value:
                return False
    return True


@njit(cache=True)
def _solve(grid: np.ndarray) -> bool:
    row, col = _find_empty(grid)
    if row == -1:
        return True
    for value in range(1, GRID_SIZE + 1):
        if _is_valid(grid, row, col, value):
            grid[row, col] = value
            if _solve(grid):
                return True
            grid[row, col] = 0
    return False


@dataclass(frozen=True)
class SudokuResult:
    solved: bool
    grid: np.ndarray


def parse_puzzle(raw: str | Iterable[str]) -> np.ndarray:
    if isinstance(raw, str):
        tokens = [char for char in raw if not char.isspace()]
    else:
        tokens = [char for char in raw if not char.isspace()]

    if len(tokens) != GRID_SIZE * GRID_SIZE:
        raise ValueError("Puzzle must contain exactly 81 digits (or dots).")

    values = []
    for char in tokens:
        if char in {"0", "."}:
            values.append(0)
        elif char.isdigit() and "1" <= char <= "9":
            values.append(int(char))
        else:
            raise ValueError("Puzzle can only contain digits 1-9, 0, or .")

    return np.array(values, dtype=np.int64).reshape((GRID_SIZE, GRID_SIZE))


def solve_sudoku(puzzle: np.ndarray) -> SudokuResult:
    grid = np.array(puzzle, dtype=np.int64, copy=True)
    solved = _solve(grid)
    return SudokuResult(solved=bool(solved), grid=grid)
