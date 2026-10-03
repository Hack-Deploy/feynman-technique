"""Discovery Market – data loading utilities.

Loads the three CSV tables and the YAML config, returning plain dicts
and DataFrames that the rest of the code relies on.
"""

import csv
import os
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
ROOT_DIR = Path(__file__).resolve().parent


def load_table1(path: Path | None = None) -> pd.DataFrame:
    """Load Table 1: mean explanation score per world per model."""
    p = path or DATA_DIR / "table1_explanation_scores.csv"
    df = pd.read_csv(p)
    df = df.set_index("world")
    return df


def load_table2(path: Path | None = None) -> pd.DataFrame:
    """Load Table 2: per-model mean_score and pass_at_1."""
    p = path or DATA_DIR / "table2_model_stats.csv"
    df = pd.read_csv(p)
    df = df.set_index("model")
    return df


def load_table3(path: Path | None = None) -> pd.DataFrame:
    """Load Table 3: numeric pass rate after 8 experiments (MDA paper)."""
    p = path or DATA_DIR / "table3_mda_pass_rates.csv"
    df = pd.read_csv(p)
    df = df.set_index("world")
    return df


# success_table.csv uses display names; map them to the Table 1 names.
SUCCESS_TABLE_MODELS = {
    "Claude Opus 4.7": "opus-4.7",
    "GPT-5.5": "gpt-5.5",
    "Claude Sonnet 4.6": "sonnet-4.6",
    "Qwen3.5-397B": "qwen3.5-397b",
}
SUCCESS_TABLE_WORLDS = {"Extra dims": "extra_dimensions"}


def load_success_table(path: Path | None = None) -> pd.DataFrame:
    """Load success_table.csv as a world x model table of pass probabilities.

    Same shape and names as Table 1, so it can stand in for it.
    """
    p = path or DATA_DIR / "success_table.csv"
    df = pd.read_csv(p)
    df["model"] = df["model"].map(SUCCESS_TABLE_MODELS)
    df["world"] = df["world"].map(
        lambda w: SUCCESS_TABLE_WORLDS.get(w, w.lower().replace(" ", "_"))
    )
    return df.pivot(index="world", columns="model", values="pass_prob")


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load config.yaml without PyYAML – simple key-value + list parser."""
    p = path or ROOT_DIR / "config.yaml"
    return _parse_yaml(p)


def _parse_yaml(path: Path) -> dict[str, Any]:
    """Minimal YAML parser sufficient for our config.yaml structure.

    Supports: scalars, lists (inline [...] and block -), nested dicts.
    Does NOT support anchors, aliases, multi-line strings, etc.
    """
    with open(path) as f:
        lines = f.readlines()

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict]] = [(-1, root)]  # (indent, dict)

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()
        i += 1

        # Skip blanks and comments
        if not stripped or stripped.lstrip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip())
        content = stripped.lstrip()

        # Pop stack to find parent
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        current_dict = stack[-1][1]

        if ":" in content:
            colon_pos = content.index(":")
            key = content[:colon_pos].strip()
            value_str = content[colon_pos + 1:].strip()

            # Remove inline comments (but not inside strings or brackets)
            if value_str and not value_str.startswith("["):
                comment_pos = value_str.find("  #")
                if comment_pos >= 0:
                    value_str = value_str[:comment_pos].strip()

            if not value_str:
                # Could be a nested dict or block list – peek ahead
                next_i = i
                while next_i < len(lines):
                    nl = lines[next_i].rstrip()
                    if nl.strip() and not nl.lstrip().startswith("#"):
                        break
                    next_i += 1

                if next_i < len(lines):
                    next_stripped = lines[next_i].lstrip()
                    if next_stripped.startswith("- "):
                        # Block list
                        lst: list[Any] = []
                        while i < len(lines):
                            nl = lines[i].rstrip()
                            if not nl.strip() or nl.lstrip().startswith("#"):
                                i += 1
                                continue
                            ni = len(lines[i]) - len(lines[i].lstrip())
                            if ni <= indent:
                                break
                            nc = nl.lstrip()
                            if nc.startswith("- "):
                                lst.append(_parse_scalar(nc[2:].strip()))
                                i += 1
                            else:
                                break
                        current_dict[key] = lst
                        continue
                    else:
                        # Nested dict
                        nested: dict[str, Any] = {}
                        current_dict[key] = nested
                        stack.append((indent, nested))
                        continue
                else:
                    current_dict[key] = None
                    continue
            elif value_str.startswith("[") and value_str.endswith("]"):
                # Inline list
                items = value_str[1:-1].split(",")
                current_dict[key] = [_parse_scalar(it.strip()) for it in items if it.strip()]
            else:
                current_dict[key] = _parse_scalar(value_str)
        elif content.startswith("- "):
            # Shouldn't happen at top level with our format
            pass

    return root


def _parse_scalar(s: str) -> int | float | str | bool | None:
    """Parse a YAML scalar value."""
    if s in ("null", "~", ""):
        return None
    if s in ("true", "True", "yes"):
        return True
    if s in ("false", "False", "no"):
        return False
    # Try int
    try:
        return int(s)
    except ValueError:
        pass
    # Try float
    try:
        return float(s)
    except ValueError:
        pass
    # Strip quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s
