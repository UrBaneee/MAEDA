"""
Debug utilities for LangGraph state inspection.

Usage:
    from src.utils.debug import diff_state, summarise_state

    # In a node or test:
    before = copy.deepcopy(state)
    state = some_node(state)
    diff_state(before, state, label="after_analysis_node")
"""
from __future__ import annotations

import copy
import json
import pprint
from typing import Any


def diff_state(before: dict, after: dict, label: str = "", print_output: bool = True) -> dict:
    """
    Show what changed in MAEDAState between two points in the graph.

    Returns a dict with keys:
      added     - keys present in after but not before
      removed   - keys present in before but not after
      modified  - keys whose values changed (shows old/new for scalars, length for lists)

    Args:
        before: State snapshot before the node ran.
        after:  State snapshot after the node ran.
        label:  Optional description printed in the header.
        print_output: If True (default), pretty-print the diff to stdout.
    """
    added: dict[str, Any] = {}
    removed: dict[str, Any] = {}
    modified: dict[str, Any] = {}

    all_keys = set(before) | set(after)
    for key in sorted(all_keys):
        in_before = key in before
        in_after = key in after

        if in_after and not in_before:
            added[key] = _summarise(after[key])
        elif in_before and not in_after:
            removed[key] = _summarise(before[key])
        else:
            bv, av = before[key], after[key]
            if bv != av:
                modified[key] = {
                    "before": _summarise(bv),
                    "after": _summarise(av),
                }

    result = {"added": added, "removed": removed, "modified": modified}

    if print_output:
        header = f"── diff_state: {label} " if label else "── diff_state "
        print(header + "─" * max(0, 60 - len(header)))
        if added:
            print("  ADDED:")
            for k, v in added.items():
                print(f"    + {k}: {v}")
        if removed:
            print("  REMOVED:")
            for k, v in removed.items():
                print(f"    - {k}: {v}")
        if modified:
            print("  MODIFIED:")
            for k, v in modified.items():
                print(f"    ~ {k}")
                print(f"        before: {v['before']}")
                print(f"        after:  {v['after']}")
        if not added and not removed and not modified:
            print("  (no changes)")
        print("─" * 60)

    return result


def summarise_state(state: dict, label: str = "") -> None:
    """
    Print a compact one-line-per-key summary of a MAEDAState snapshot.
    Useful at the start/end of a test to inspect full state at a glance.
    """
    header = f"── state: {label} " if label else "── state "
    print(header + "─" * max(0, 60 - len(header)))
    for key in sorted(state):
        print(f"  {key:30s} {_summarise(state[key])}")
    print("─" * 60)


def _summarise(value: Any) -> str:
    """Compact representation of a value for diff output."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        if len(value) <= 80:
            return repr(value)
        return repr(value[:77] + "...")
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return f"dict[{len(value)} keys]"
    return repr(value)[:80]
