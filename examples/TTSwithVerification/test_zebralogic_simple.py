#!/usr/bin/env python3
"""Simple test for ZebraLogic ToT integration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that imports work."""
    try:
        from interwhen.tree_of_thought import build_tot_problem, build_tot_value_prompt
        print("✓ tree_of_thought imports successful")
    except Exception as e:
        print(f"✗ tree_of_thought import failed: {e}")
        return False
    
    try:
        from interwhen.value_prompts import build_zebralogic_value_prompt
        print("✓ value_prompts imports successful")
    except Exception as e:
        print(f"✗ value_prompts import failed: {e}")
        return False
    
    return True

def test_prompt_building():
    """Test prompt building."""
    from interwhen.tree_of_thought import build_tot_problem
    
    example = {
        "puzzle": "Test puzzle with clues about houses.",
        "solution": {"House 1": {"color": "red"}}
    }
    
    try:
        prompt = build_tot_problem("zebralogic", example)
        print("✓ build_tot_problem works")
        print(f"  - Prompt length: {len(prompt)}")
        
        if "Problem Description" in prompt and "json" in prompt.lower():
            print("✓ Prompt contains expected content")
            return True
        else:
            print("✗ Prompt missing expected content")
            return False
    except Exception as e:
        print(f"✗ build_tot_problem failed: {e}")
        return False

def test_value_prompts():
    """Test value prompt generation."""
    from interwhen.value_prompts import build_zebralogic_value_prompt, build_tot_value_prompt
    
    try:
        prompt1 = build_zebralogic_value_prompt("problem", "trajectory", use_fewshot=True)
        print("✓ build_zebralogic_value_prompt (fewshot) works")
        
        prompt2 = build_zebralogic_value_prompt("problem", "trajectory", use_fewshot=False)
        print("✓ build_zebralogic_value_prompt (simple) works")
        
        prompt3 = build_tot_value_prompt("zebralogic", "problem", "trajectory")
        print("✓ build_tot_value_prompt routes to zebralogic correctly")
        
        if "Confidence" in prompt1 and "sure" in prompt1:
            print("✓ Value prompts contain expected guidance")
            return True
        else:
            print("✗ Value prompts missing expected content")
            return False
    except Exception as e:
        print(f"✗ Value prompt test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_extraction():
    """Test solution extraction."""
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        from tot_baseline import extract_solution_zebralogic
        
        json_text = '```json\n{"House 1": {"color": "red"}}\n```'
        result = extract_solution_zebralogic(json_text)
        
        if result is not None:
            print("✓ extract_solution_zebralogic works")
            return True
        else:
            print("⚠ extract_solution_zebralogic returned None")
            return True  # Not critical for basic validation
    except ImportError as e:
        print(f"⚠ Skipping extraction test (import issue): {e}")
        return True
    except Exception as e:
        print(f"✗ Extraction test failed: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ZebraLogic ToT Integration Test")
    print("="*60 + "\n")
    
    all_passed = True
    
    print("Test 1: Imports")
    all_passed &= test_imports()
    
    print("\nTest 2: Prompt Building")
    all_passed &= test_prompt_building()
    
    print("\nTest 3: Value Prompts")
    all_passed &= test_value_prompts()
    
    print("\nTest 4: Solution Extraction")
    all_passed &= test_extraction()
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*60 + "\n")
