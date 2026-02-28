#!/usr/bin/env python3
"""Command-line Tree-of-Thought baseline runner for interwhen datasets."""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bestofk_baseline import (
    build_maze_prompt,
    build_spatialmap_prompt,
    evaluate_game24_answer,
    evaluate_mcq_answer,
    extract_options_from_prompt,
    load_dataset_for_task,
    resolve_indices,
)

from interwhen.tree_of_thought import (
    SearchMethod,
    ToTSearchConfig,
    TreeOfThoughtSearch,
    build_tot_problem,
)

LOGGER = logging.getLogger("tot_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tree-of-Thought search on a subset of the supported tasks",
    )
    parser.add_argument("--task", choices=["game24", "maze", "spatialmap"], required=True)
    parser.add_argument("--k", type=int, default=1, help="Unused placeholder to mirror other baselines")
    parser.add_argument("--num_examples", "-n", type=int, default=None)
    parser.add_argument("--indices", type=str, default=None)
    parser.add_argument("--xrange", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--model", default="Qwen/QwQ-32B")
    parser.add_argument("--llm_url", default="http://localhost:{port}/v1/chat/completions")
    parser.add_argument(
        "--ports",
        default="8000",
        help="Comma-separated list of vLLM ports to round-robin across",
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search_method", choices=["bfs", "dfs", "beam"], default="bfs")
    parser.add_argument("--branching_factor", type=int, default=4)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--beam_width", type=int, default=2)
    parser.add_argument("--sure_threshold", type=float, default=0.7)
    parser.add_argument("--likely_threshold", type=float, default=0.5)
    parser.add_argument("--impossible_threshold", type=float, default=0.2)
    parser.add_argument("--max_candidates_per_level", type=int, default=3)
    parser.add_argument("--early_termination", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of ToT examples to run concurrently",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/tot_baseline",
        help="Directory to store per-example JSON logs and summary",
    )
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def parse_port_list(port_str: str) -> List[int]:
    return [int(p.strip()) for p in port_str.split(",") if p.strip()]


def build_llm_server(args: argparse.Namespace, port: int) -> Dict[str, Any]:
    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "stream": False,
        "seed": args.seed,
    }
    return {
        "url": args.llm_url.format(port=port),
        "headers": {"content-type": "application/json"},
        "payload": payload,
    }


def build_tot_config(args: argparse.Namespace) -> ToTSearchConfig:
    method = SearchMethod[args.search_method.upper()]
    return ToTSearchConfig(
        branching_factor=args.branching_factor,
        max_depth=args.max_depth,
        search_method=method,
        beam_width=args.beam_width,
        sure_threshold=args.sure_threshold,
        likely_threshold=args.likely_threshold,
        impossible_threshold=args.impossible_threshold,
        early_termination=args.early_termination,
        cache_evaluations=not args.no_cache,
        max_candidates_per_level=args.max_candidates_per_level,
    )


def ensure_output_dir(base_dir: str, task: str) -> Path:
    path = Path(base_dir).expanduser().resolve() / task
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_eval(task: str, example: Dict[str, Any]) -> Tuple:
    if task == "game24":
        nums = list(example.get("numbers", []))
        return (lambda output: evaluate_game24_answer(output, nums), {"numbers": nums})
    gt = str(example.get("ground_truth", "")).strip()
    target_options = ["A", "B"] if gt == "Q4" else ["A", "B", "C", "D"]
    if task == "maze":
        _, user_prompt = build_maze_prompt(example)
    else:
        _, user_prompt = build_spatialmap_prompt(example)
    options = extract_options_from_prompt(user_prompt, target_options)
    meta = {"options": options, "ground_truth": gt}
    return (lambda output: evaluate_mcq_answer(output, options, gt), meta)


async def run_single_example(
    idx: int,
    task: str,
    example: Dict[str, Any],
    tot_config: ToTSearchConfig,
    llm_server: Dict[str, Any],
) -> Dict[str, Any]:
    eval_fn, eval_meta = prepare_eval(task, example)
    problem = build_tot_problem(task, example, nums=example.get("numbers"))
    tot = TreeOfThoughtSearch(tot_config)
    search_result = await tot.search(problem, llm_server)
    best_traj = search_result.get("best_trajectory", "")
    best_value = search_result.get("best_value", 0.0)
    is_correct, extracted, message = eval_fn(best_traj)
    return {
        "index": int(idx),
        "best_value": best_value,
        "best_trajectory": best_traj,
        "search_stats": search_result.get("search_stats", {}),
        "correct": bool(is_correct),
        "extracted": extracted,
        "message": message,
        "evaluation_meta": eval_meta,
    }


async def run_tot_baseline(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    dataset = load_dataset_for_task(args.task)
    indices = resolve_indices(args.task, len(dataset), args)
    output_dir = ensure_output_dir(args.output_dir, args.task)
    tot_config = build_tot_config(args)
    ports = parse_port_list(args.ports)
    if not ports:
        raise ValueError("At least one port must be specified via --ports")
    concurrency = max(1, args.concurrency)
    port_lock = asyncio.Lock()
    port_index = {"value": 0}

    async def next_port() -> int:
        async with port_lock:
            port = ports[port_index["value"] % len(ports)]
            port_index["value"] += 1
            return port

    semaphore = asyncio.Semaphore(concurrency)

    async def process_index(idx: int) -> Dict[str, Any]:
        async with semaphore:
            example = dataset[int(idx)]
            port = await next_port()
            llm_server = build_llm_server(args, port)
            LOGGER.info("Running ToT on example %s via port %s", idx, port)
            try:
                record = await run_single_example(idx, args.task, example, tot_config, llm_server)
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("Failed example %s", idx)
                record = {
                    "index": int(idx),
                    "error": str(exc),
                    "best_trajectory": "",
                    "correct": False,
                }
            example_path = output_dir / f"example_{idx}.json"
            with example_path.open("w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
            return record

    processed = await asyncio.gather(*[process_index(idx) for idx in indices])

    total = len(processed)
    correct = sum(1 for r in processed if r.get("correct"))
    summary = {
        "task": args.task,
        "model": args.model,
        "total_examples": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "search_method": args.search_method,
        "config": {
            "branching_factor": args.branching_factor,
            "max_depth": args.max_depth,
            "beam_width": args.beam_width,
            "sure_threshold": args.sure_threshold,
            "likely_threshold": args.likely_threshold,
            "impossible_threshold": args.impossible_threshold,
            "max_candidates_per_level": args.max_candidates_per_level,
            "early_termination": args.early_termination,
            "cache_evaluations": not args.no_cache,
            "ports": ports,
            "concurrency": concurrency,
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info("Accuracy %.2f (%d/%d)", summary["accuracy"], correct, total)


if __name__ == "__main__":
    asyncio.run(run_tot_baseline(parse_args()))
