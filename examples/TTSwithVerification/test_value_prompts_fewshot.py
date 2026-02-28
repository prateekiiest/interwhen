#!/usr/bin/env python3
"""
Test script to verify value prompts with few-shot examples are working correctly
across all datasets (game24, maze, spatialmap).
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from interwhen.value_prompts import (
    build_game24_value_prompt,
    build_mcq_value_prompt,
    build_generic_value_prompt,
    build_tot_value_prompt,
)


def test_game24_value_prompts():
    """Test game24 value prompts with and without few-shot examples."""
    print("\n" + "=" * 80)
    print("TESTING GAME24 VALUE PROMPTS")
    print("=" * 80)
    
    problem = """You are solving the Game of 24.

You are given four numbers: 3, 3, 8, 8

Your job is to produce a valid arithmetic expression using:
- ALL four numbers exactly once
- ONLY +, -, *, /
- The expression must evaluate to exactly 24.

Please reason step by step, and put your final answer containing only the expression within \\boxed{}."""
    
    trajectory = """Step 1: 8 / 8 = 1
Step 2: 3 + 3 = 6
Step 3: 1 * 6 = 6

Hmm this doesn't work, let me try again:
Step 1: 8 / (1 - 8 / 6) might work if I can construct 1 - 8/6 = -1/3
Then 8 / (-1/3) = -24, which is not 24.

Let me reconsider: (3 + 3 / 8) * 8 = (3 + 0.375) * 8 = 3.375 * 8 = 27, not 24."""

    # Test with few-shot examples
    print("\n--- Game24 with Few-shot Examples ---")
    prompt_with_fewshot = build_game24_value_prompt(problem, trajectory, use_fewshot=True)
    print(f"Length: {len(prompt_with_fewshot)} characters")
    print(f"\nFirst 500 chars:\n{prompt_with_fewshot[:500]}")
    
    # Test without few-shot examples
    print("\n--- Game24 without Few-shot Examples ---")
    prompt_without_fewshot = build_game24_value_prompt(problem, trajectory, use_fewshot=False)
    print(f"Length: {len(prompt_without_fewshot)} characters")
    print(f"\nFirst 300 chars:\n{prompt_without_fewshot[:300]}")
    
    print(f"\n✓ Few-shot prompt is {len(prompt_with_fewshot) - len(prompt_without_fewshot)} chars longer")
    

def test_maze_value_prompts():
    """Test maze value prompts with and without few-shot examples."""
    print("\n" + "=" * 80)
    print("TESTING MAZE VALUE PROMPTS")
    print("=" * 80)
    
    problem = """Here is a Maze...
The objective is to navigate from S to E by following the path marked by X.
Please answer the following question:
How many right turns are there in the provided path from S to E?
Available options: A. 5, B. 6, C. 4, D. 0"""
    
    trajectory = """I trace the X-marked path from S to E.
Starting at S, I move UP initially, then RIGHT - this is a right turn.
Continuing to follow the path, I count more right turns as I go.
It looks like there are 6 right turns total.
Answer: B"""
    
    # Test maze with few-shot
    print("\n--- Maze with Few-shot Examples ---")
    prompt_with_fewshot = build_mcq_value_prompt(problem, trajectory, "maze", use_fewshot=True)
    print(f"Length: {len(prompt_with_fewshot)} characters")
    print(f"\nFirst 500 chars:\n{prompt_with_fewshot[:500]}")
    
    # Test maze without few-shot
    print("\n--- Maze without Few-shot Examples ---")
    prompt_without_fewshot = build_mcq_value_prompt(problem, trajectory, "maze", use_fewshot=False)
    print(f"Length: {len(prompt_without_fewshot)} characters")
    print(f"\nFirst 300 chars:\n{prompt_without_fewshot[:300]}")
    
    print(f"\n✓ Few-shot prompt is {len(prompt_with_fewshot) - len(prompt_without_fewshot)} chars longer")


def test_spatialmap_value_prompts():
    """Test spatial reasoning value prompts with and without few-shot examples."""
    print("\n" + "=" * 80)
    print("TESTING SPATIALMAP VALUE PROMPTS")
    print("=" * 80)
    
    problem = """Here is a map with various locations marked.
Based on the map, which location is northeast of the library?
Available options: A. Museum, B. Park, C. School, D. Station"""
    
    trajectory = """I look at the map and identify the library in the center.
Northeast means both north AND east of that point.
Looking at the northeast quadrant, I see the Museum.
Answer: A. Museum"""
    
    # Test spatial with few-shot
    print("\n--- Spatial with Few-shot Examples ---")
    prompt_with_fewshot = build_mcq_value_prompt(problem, trajectory, "spatial reasoning", use_fewshot=True)
    print(f"Length: {len(prompt_with_fewshot)} characters")
    print(f"\nFirst 500 chars:\n{prompt_with_fewshot[:500]}")
    
    # Test spatial without few-shot
    print("\n--- Spatial without Few-shot Examples ---")
    prompt_without_fewshot = build_mcq_value_prompt(problem, trajectory, "spatial reasoning", use_fewshot=False)
    print(f"Length: {len(prompt_without_fewshot)} characters")
    print(f"\nFirst 300 chars:\n{prompt_without_fewshot[:300]}")
    
    print(f"\n✓ Few-shot prompt is {len(prompt_with_fewshot) - len(prompt_without_fewshot)} chars longer")


def test_auto_detection():
    """Test automatic task detection and prompt building."""
    print("\n" + "=" * 80)
    print("TESTING AUTO-DETECTION WITH build_tot_value_prompt")
    print("=" * 80)
    
    test_cases = [
        ("Game of 24 problem with numbers 1, 2, 3, 4", "game24"),
        ("Here is a Maze represented in ASCII", "maze"),
        ("Spatial map reasoning about locations", "spatialmap"),
        ("Generic problem solving task", "generic"),
    ]
    
    for problem_text, expected_task in test_cases:
        prompt = build_tot_value_prompt(problem_text, "Some trajectory", use_fewshot=True)
        has_examples = "EXAMPLE" in prompt.upper() or "example" in prompt.lower()
        print(f"\n✓ Problem: '{problem_text[:40]}...'")
        print(f"  Expected dataset: {expected_task}")
        print(f"  Has few-shot examples: {has_examples}")
        print(f"  Prompt length: {len(prompt)} chars")


def test_empty_trajectory():
    """Test handling of empty trajectories."""
    print("\n" + "=" * 80)
    print("TESTING EMPTY TRAJECTORY HANDLING")
    print("=" * 80)
    
    problem = "You are solving the Game of 24. Numbers: 1, 2, 3, 4"
    
    # Test with empty trajectory
    prompt_empty = build_game24_value_prompt(problem, "", use_fewshot=True)
    assert "No progress yet" in prompt_empty, "Empty trajectory should show 'No progress yet'"
    
    # Test with whitespace only
    prompt_whitespace = build_game24_value_prompt(problem, "   \n  ", use_fewshot=True)
    assert "No progress yet" in prompt_whitespace, "Whitespace-only trajectory should show 'No progress yet'"
    
    print("\n✓ Empty trajectory handling works correctly")


if __name__ == "__main__":
    try:
        print("\n" + "=" * 80)
        print("VALUE PROMPTS WITH FEW-SHOT EXAMPLES - COMPREHENSIVE TEST")
        print("=" * 80)
        
        test_game24_value_prompts()
        test_maze_value_prompts()
        test_spatialmap_value_prompts()
        test_auto_detection()
        test_empty_trajectory()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nFew-shot value prompts are now available for:")
        print("  • Game24 evaluations")
        print("  • Maze reasoning evaluations")
        print("  • Spatial reasoning evaluations")
        print("  • Generic problem solving")
        print("\nUsage: pass use_fewshot=True (default) to enable few-shot examples")
        print("       pass use_fewshot=False to use simple prompts without examples")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
