import os
import time
from enum import Enum

import imageio.v2 as imageio
import numpy as np
import torch
import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from conway import BG, OG, evolve, get_border

app = typer.Typer()
console = Console()

ALIVE_CHAR = "O"
DEAD_CHAR = "\u00b7"
PADDING = 2


class SeedPattern(Enum):
    GLIDERS = "0,0 1,1 2,-1 2,0 2,1 10,5 11,6 12,4 12,5 12,6 20,10 21,11 22,9 22,10 22,11"
    BLINKER = "0,0 1,0 2,0"
    TOAD = "0,1 1,0 1,1 2,1 3,0 3,1"
    BEACON = "0,0 1,0 2,0 2,1 3,1 3,2 4,2 4,3 5,2 5,3"
    PULSAR = "0,1 0,2 0,3 1,-1 1,4 2,-1 2,4 3,0 3,1 3,2 3,3 4,-1 4,4 5,-1 5,4 6,1 6,2 6,3"
    GOSPER_GUN = (
        "0,0 1,0 0,1 1,1 "
        "10,-1 14,-1 10,0 11,0 12,0 13,0 14,0 15,0 16,0 17,0 20,-1 20,0 20,1 21,-2 21,2 "
        "22,-3 22,3 23,-1 23,0 23,1 24,0 "
        "34,-2 34,-1 35,-2 35,-1"
    )

    @property
    def coordinates(self) -> str:
        return self.value


def render_frame(
    og: OG, max_coord: int, offset: torch.Tensor | None = None
) -> Text:
    if og.nodes.numel() == 0:
        return Text("Empty grid", style="bold red")

    if offset is None:
        center = og.nodes.float().mean(dim=0)
        offset = torch.round(center).to(torch.int64)

    alive_set = {tuple(node.tolist()) for node in (og.nodes - offset)}

    rows = []
    for y in range(max_coord - 1, -max_coord, -1):
        row = "".join(
            ALIVE_CHAR if (x, y) in alive_set else DEAD_CHAR
            for x in range(-max_coord, max_coord + 1)
        )
        rows.append(row)

    grid_text = "\n".join(rows)
    return Text(grid_text, style="bold green")


def frame_to_image(
    og: OG,
    max_coord: int,
    offset: torch.Tensor | None = None,
    pixel_size: int = 12,
) -> np.ndarray:
    if og.nodes.numel() == 0:
        return np.zeros((2 * max_coord + 1, 2 * max_coord + 1), dtype=np.uint8)

    if offset is None:
        center = og.nodes.float().mean(dim=0)
        offset = torch.round(center).to(torch.int64)

    alive_set = {tuple(node.tolist()) for node in (og.nodes - offset)}

    rows = []
    for y in range(max_coord - 1, -max_coord, -1):
        row = np.array(
            [
                255 if (x, y) in alive_set else 0
                for x in range(-max_coord, max_coord + 1)
            ],
            dtype=np.uint8,
        )
        rows.append(row)

    grid = np.array(rows, dtype=np.uint8)
    return np.kron(grid, np.ones((pixel_size, pixel_size), dtype=np.uint8))


@app.command()
def main(
    seed: SeedPattern = typer.Option(
        SeedPattern.GLIDERS,
        "--seed",
        "-s",
        help=f"Seed pattern: {', '.join(p.name for p in SeedPattern)} or custom coordinates",
    ),
    iterations: int = typer.Option(
        2**31 - 1,
        "--iterations",
        "-n",
        help="Number of iterations (default: run forever)",
    ),
    delay: float = typer.Option(
        0.1, "--delay", "-d", help="Delay between iterations (seconds)"
    ),
    grid_size: int = typer.Option(
        20, "--grid-size", "-g", help="Half-size of visible grid"
    ),
    gif: str | None = typer.Option(
        None, "--gif", "-f", help="Filename to save GIF (saved in figures/)"
    ),
    random: bool = typer.Option(
        False, "--random", "-r", help="Use random seed nodes"
    ),
) -> None:
    seed_coords = seed.coordinates if isinstance(seed, SeedPattern) else seed
    seed_nodes = torch.tensor(
        [
            [int(x), int(y)]
            for pair in seed_coords.split()
            for x, y in [pair.split(",")]
        ]
    )

    og = OG(nodes=seed_nodes)
    bg = BG(nodes=get_border(og))

    center = seed_nodes.float().mean(dim=0)
    offset = torch.round(center).to(torch.int64)

    frames: list = []
    if gif:
        os.makedirs("figures", exist_ok=True)
        frames.append(frame_to_image(og, grid_size, offset))

    pattern_name = seed.name if isinstance(seed, SeedPattern) else "Custom"
    with Live(
        Panel(
            render_frame(og, grid_size, offset),
            title=f"Conway's Game of Life - {pattern_name}",
            border_style="blue",
        ),
        console=console,
        refresh_per_second=10,
        transient=False,
    ) as live:
        for i in range(iterations):
            og, bg = evolve(og, bg, random=random)
            if gif:
                frames.append(frame_to_image(og, grid_size, offset))
            live.update(
                Panel(
                    render_frame(og, grid_size, offset),
                    title=f"Conway's Game of Life - {pattern_name} (iter {i + 1})",
                    border_style="blue",
                )
            )
            time.sleep(delay)

    if gif:
        path = os.path.join("figures", gif)
        if not gif.endswith(".gif"):
            path += ".gif"
        imageio.mimsave(path, frames, fps=int(1 / delay))
        console.print(f"[bold green]GIF saved to {path}[/bold green]")


if __name__ == "__main__":
    app()
