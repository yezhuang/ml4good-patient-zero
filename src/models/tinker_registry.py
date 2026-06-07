"""Resolve Tinker checkpoints by trait/pole from a markdown registry.

The registry is a markdown table with `trait | pole | tinker:// URI` rows, e.g.
`configs/checkpoints/qwen8b_base_sft_epoch10.md`. Configs reference a checkpoint
by `"trait/pole"` (e.g. `"power-seeking/plus"`) instead of pasting the URI.

These URIs are account-specific, so the registry file is gitignored.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_REGISTRY_PATH = Path("configs/checkpoints/qwen8b_base_sft_epoch10.md")


def _key(trait: str, pole: str) -> str:
    return f"{trait.strip().lower()}/{pole.strip().lower()}"


def load_checkpoint_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, str]:
    """Parse a markdown table into a {"trait/pole": "tinker://..."} mapping."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint registry not found: {path}. Copy the checkpoints markdown "
            f"into {DEFAULT_REGISTRY_PATH} or set 'checkpoints_file' in the config."
        )

    registry: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        trait, pole, uri = cells[0], cells[1], cells[2].strip("`").strip()
        if not uri.startswith("tinker://"):
            continue  # skips the header and separator rows
        registry[_key(trait, pole)] = uri

    if not registry:
        raise ValueError(f"No tinker:// checkpoints parsed from {path}")
    return registry


def resolve_checkpoint(
    ref: str | None = None,
    *,
    path: str | Path = DEFAULT_REGISTRY_PATH,
    trait: str | None = None,
    pole: str | None = None,
) -> str:
    """Look up a checkpoint URI by `"trait/pole"` ref or explicit trait+pole."""
    registry = load_checkpoint_registry(path)

    if ref is not None:
        key = ref.strip().lower().replace(":", "/")
    elif trait and pole:
        key = _key(trait, pole)
    else:
        raise ValueError("Provide either 'ref' or both 'trait' and 'pole'.")

    if key not in registry:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Checkpoint '{key}' not in registry. Available: {available}")
    return registry[key]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="List Tinker checkpoint refs.")
    parser.add_argument(
        "--path", default=str(DEFAULT_REGISTRY_PATH), help="Registry markdown path."
    )
    parser.add_argument("--uris", action="store_true", help="Also print the URIs.")
    args = parser.parse_args()

    registry = load_checkpoint_registry(args.path)
    for key in sorted(registry):
        print(f"{key}\t{registry[key]}" if args.uris else key)
    print(f"\n{len(registry)} checkpoints", flush=True)


if __name__ == "__main__":
    main()
