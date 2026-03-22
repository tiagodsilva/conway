# Agent Guidelines for Conway

## Project Overview

This is a Python implementation of Conway's Game of Life using PyTorch. The project uses an infinite graph representation rather than a fixed matrix.

## Commands

### Package Management (uv)
```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package>

# Remove a dependency
uv remove <package>
```

### Type Checking (pyright)
```bash
# Run type checking
pyright

# Run on specific file
pyright main.py
```

### Python Execution
```bash
# Run the main module
python main.py

# Run with specific Python
python3.13 main.py
```

### Testing
This project does not currently have a test suite. When adding tests:
```bash
# Install pytest
uv add pytest

# Run all tests
pytest

# Run a single test file
pytest tests/test_main.py

# Run a specific test
pytest tests/test_main.py::test_function_name

# Run tests matching a pattern
pytest -k "test_pattern"
```

## Code Style Guidelines

### General
- Target Python 3.13+
- Use type hints for all function parameters and return values
- Keep functions focused and small (prefer single responsibility)

### Imports
- Standard library imports first
- Third-party imports (torch, numpy, typer, rich) second
- Local/relative imports last
- Separate groups with blank lines
- Sort imports alphabetically within groups
```python
from dataclasses import dataclass, replace

import torch
```

### Formatting
- 4 spaces for indentation
- Maximum line length: 100 characters (soft)
- Use vertical spacing to group related logic
- Prefer explicit line breaks over backslash continuations

### Naming Conventions
- **Classes**: PascalCase (`OG`, `BG`)
- **Functions/methods**: snake_case (`count_neighbors`, `get_border`)
- **Variables**: snake_case (`n_neigh_bg`, `remaining_nodes`)
- **Constants**: SCREAMING_SNAKE_CASE (e.g., `ALIVE_CHAR`)
- **Type variables**: PascalCase (e.g., `T`, `NodeT`)

### Type Annotations
- Use `torch.Tensor` for tensor types
- Use dataclasses for structured data containers
- Prefer explicit return type annotations
```python
def count_neighbors(new_nodes: torch.Tensor, curr_nodes: torch.Tensor) -> torch.Tensor:
```

### Dataclasses
- Use `@dataclass` for simple data containers
- Use `replace()` to create modified copies (immutable pattern)
```python
@dataclass
class OG:
    nodes: torch.Tensor  # (N, 2)

og = replace(og, nodes=new_nodes)
```

### Comments
- Use comments sparingly; prefer self-documenting code
- Complex algorithms may benefit from explanatory comments
- Keep comments up-to-date with code changes
- Use inline comments sparingly for non-obvious logic

### Error Handling
- Use explicit type checking rather than try/except for type errors
- Let PyTorch handle numerical errors (NaN, inf) naturally
- Prefer validation at function entry points

### Tensor Operations
- Prefer in-place operations when memory is a concern
- Use descriptive variable names for tensor shapes in comments
- Document tensor shapes in function signatures and key variables
- **Important**: A cell does NOT count itself as a neighbor (use `distance == 0` to exclude self)
- Use 8-connected neighbors (Chebyshev distance <= 1) for Conway's Game of Life

## Project Structure

```
conway/
├── main.py          # Core Game of Life logic
├── visualize.py     # Terminal-based visualization (Typer CLI)
├── pyproject.toml   # Project configuration
├── pyrightconfig.json
├── .venv/           # Virtual environment (do not commit)
└── README.md        # Project description
```

### Visualization Usage (Typer CLI)

```bash
# Interactive mode (runs forever by default, Ctrl+C to quit)
python visualize.py main --delay 0.1 --grid-size 20

# Batch mode (prints iterations to stdout)
python visualize.py batch --iterations 20 --print-every 2

# Custom seed (x,y coordinate pairs)
python visualize.py main --seed "0,0 1,0 2,0 1,1"

# All options
python visualize.py --help
python visualize.py main --help
python visualize.py batch --help
```

**CLI Options:**
- `--seed / -s`: Coordinate pairs (default: three gliders)
- `--iterations / -n`: Number of iterations (default: 2147483647 / run forever in main mode)
- `--delay / -d`: Delay between frames in seconds (default: 0.1)
- `--grid-size / -g`: Half-size of visible grid (default: 20)
- `--print-every / -p`: Print every N iterations (batch mode, default: 1)

## Common Patterns

### Tensor Shape Conventions
- Single coordinates: `(2,)` - a single [x, y] pair
- Multiple coordinates: `(N, 2)` - N coordinate pairs
- Distance matrices: `(K, N)` - distances from K candidates to N nodes

### Graph Operations
- OG (Original Graph): The living cells
- BG (Border Graph): Potential cells that could become alive
- Border is computed as nodes at Chebyshev distance 1 from OG that aren't already in OG

### Conway's Game of Life Rules
1. A cell with < 2 neighbors dies (underpopulation)
2. A cell with 2 or 3 neighbors survives
3. A cell with > 3 neighbors dies (overpopulation)
4. A dead cell with exactly 3 neighbors becomes alive (reproduction)
