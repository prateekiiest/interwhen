"""
Tree of Thought implementation for interwhen-style streaming completion.

Implements proper ToT search using:
1. Propose function to generate candidate next steps
2. Value function to evaluate intermediate states
3. Search algorithm (BFS/DFS/beam) to explore the tree
4. Integrated with interwhen's async streaming architecture
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

from .value_prompts import (
    build_game24_value_prompt,
    build_mcq_value_prompt,
    build_generic_value_prompt,
    build_tot_value_prompt as build_tot_value_prompt_impl,
    _detect_tot_task,
)

logger = logging.getLogger(__name__)


# --------------------- Dataset prompt helpers ---------------------

def remove_last_paragraph(text: str) -> str:
    """Placeholder hook mirroring the TTS baselines (kept for parity)."""
    return text


def build_game24_prompt(nums: List[int]) -> str:
    """Return the canonical Game24 instruction block used across baselines."""
    if len(nums) != 4:
        raise ValueError("Game24 requires exactly four numbers.")
    a, b, c, d = nums
    boxed = r"\\boxed{}"
    return (
        "You are solving the Game of 24.\n\n"
        f"You are given four numbers: {a}, {b}, {c}, {d}\n\n"
        "Your job is to produce a valid arithmetic expression using:\n"
        "- ALL four numbers exactly once\n- ONLY +, -, *, /\n"
        "- The expression must evaluate to exactly 24.\n\n"
        "Please reason step by step, and put your final answer containing"
        f" only the expression within {boxed}."
    )


def build_maze_prompt(example: Dict[str, Any]) -> str:
    """Construct the maze reasoning instructions used in other pipelines."""
    pre_prompt = (
        "You are an expert problem solver. Carefully read the following "
        "multiple-choice question and think through the solution step-by-step "
        "before providing your final answer. Provide the final answer option by "
        "enclosing it within \\boxed{A/B/C/D}."
    )
    description = remove_last_paragraph(str(example.get("prompt", "")))
    return f"{pre_prompt}\n\n{description.strip()}"


def build_spatialmap_prompt(example: Dict[str, Any]) -> str:
    """Construct the spatial reasoning instructions for TOT experiments."""
    pre_prompt = (
        "You are an expert problem solver. Carefully read the following "
        "multiple-choice question and think through the solution step-by-step "
        "before providing your final answer. Provide the final answer option by "
        "enclosing it within \\boxed{A/B/C/D}."
    )
    description = remove_last_paragraph(str(example.get("prompt", "")))
    return f"{pre_prompt}\n\n{description.strip()}"


def build_tot_problem(task: str, example: Dict[str, Any], nums: Optional[List[int]] = None) -> str:
    """Helper that mirrors the best-of-k prompt builders for ToT runs."""
    task_lower = task.lower()
    if task_lower == "game24":
        numbers = nums or example.get("numbers")
        if numbers is None:
            raise ValueError("Game24 prompt requires 'numbers' in the example")
        return build_game24_prompt(list(numbers))
    if task_lower == "maze":
        return build_maze_prompt(example)
    if task_lower == "spatialmap":
        return build_spatialmap_prompt(example)
    raise ValueError(f"Unsupported task for ToT prompt building: {task}")


def build_tot_value_prompt(
    problem: str,
    trajectory: str,
    use_fewshot: bool = True
) -> str:
    """
    Build value prompt for Tree of Thought evaluation.
    
    Args:
        problem: The original problem statement
        trajectory: Current partial solution (or 'No progress yet' if empty)
        use_fewshot: Whether to use few-shot examples (default True for better evaluation)
        
    Returns:
        Formatted value prompt with or without few-shot examples
    """
    if not trajectory.strip():
        trajectory = "No progress yet"
    return build_tot_value_prompt_impl(problem, trajectory, use_fewshot=use_fewshot)


class SearchMethod(Enum):
    """Search algorithm types"""
    BFS = "bfs"
    DFS = "dfs"
    BEAM = "beam"


@dataclass
class TreeNode:
    """Represents a node in the Tree of Thought"""
    trajectory: str
    depth: int
    value: float = 0.5
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    is_terminal: bool = False
    proposals: List[str] = field(default_factory=list)
    evaluation_log: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.trajectory)
    
    def __eq__(self, other):
        return isinstance(other, TreeNode) and self.trajectory == other.trajectory


@dataclass
class ToTSearchConfig:
    """Configuration for Tree of Thought search"""
    branching_factor: int = 4
    max_depth: int = 6
    search_method: SearchMethod = SearchMethod.BFS
    beam_width: int = 2
    
    # Value thresholds
    sure_threshold: float = 0.7
    likely_threshold: float = 0.5
    impossible_threshold: float = 0.2
    
    # Optimization settings
    early_termination: bool = True
    cache_evaluations: bool = True
    max_candidates_per_level: int = 3


class TreeOfThoughtSearch:
    """
    Tree of Thought search controller compatible with interwhen's streaming.
    
    Provides propose/evaluate/search methods that work with vLLM API calls
    via the llm_server interface used in interwhen.
    """
    
    def __init__(self, config: ToTSearchConfig = None):
        self.config = config or ToTSearchConfig()
        self.evaluation_cache = {}
        self.proposal_cache = {}
        self.search_stats = {
            "nodes_explored": 0,
            "evaluations_performed": 0,
            "branches_pruned": 0,
            "cache_hits": 0,
            "solutions_found": 0,
            "total_nodes_in_tree": 0,
        }
        self.decision_tree = []
        self.root = None
    
    # ===================== PROPOSE FUNCTION =====================
    
    async def propose_next_steps(
        self,
        problem: str,
        current_trajectory: str,
        llm_server: Dict,
        num_proposals: Optional[int] = None,
    ) -> List[str]:
        """
        Generate candidate next steps using the model's propose capability.
        
        Args:
            problem: The original problem statement
            current_trajectory: Current partial solution
            llm_server: vLLM server config (url, headers, payload template)
            num_proposals: Number of proposals to generate (defaults to branching_factor)
            
        Returns:
            List of proposed next steps
        """
        if num_proposals is None:
            num_proposals = self.config.branching_factor
        
        # Check cache
        cache_key = f"propose_{hash(problem)}_{hash(current_trajectory)}"
        if self.config.cache_evaluations and cache_key in self.proposal_cache:
            self.search_stats["cache_hits"] += 1
            return self.proposal_cache[cache_key]
        
        self.search_stats["nodes_explored"] += 1
        
        # Build propose prompt
        propose_prompt = self._build_propose_prompt(
            problem, 
            current_trajectory, 
            num_proposals
        )
        
        # Call model with streaming
        proposal_text = await self._call_llm_streaming(
            llm_server,
            propose_prompt
        )
        
        # Parse proposals from response
        proposals = self._parse_proposals(proposal_text, num_proposals)
        
        # Log decision point
        decision_log = {
            "type": "proposal_generation",
            "timestamp": time.time(),
            "problem_hash": hash(problem),
            "trajectory": current_trajectory,
            "prompt": propose_prompt[:200] + "..." if len(propose_prompt) > 200 else propose_prompt,
            "raw_response": proposal_text[:300] + "..." if len(proposal_text) > 300 else proposal_text,
            "parsed_proposals": proposals,
        }
        self.decision_tree.append(decision_log)
        
        # Cache
        if self.config.cache_evaluations:
            self.proposal_cache[cache_key] = proposals
        
        return proposals
    
    def _build_propose_prompt(
        self,
        problem: str,
        trajectory: str,
        num_proposals: int
    ) -> str:
        """Build a prompt requesting proposals for next steps"""
        return f"""Given the following problem and current progress, propose {num_proposals} possible next steps.

PROBLEM:
{problem}

CURRENT PROGRESS/TRAJECTORY:
{trajectory if trajectory.strip() else "Starting fresh - no progress yet"}

Generate {num_proposals} distinct next steps that could advance the solution. Be specific and actionable.

Format each proposal clearly, one per line:
1. [Proposal 1]
2. [Proposal 2]
...

Think step by step about what makes each proposal viable.
"""
    
    def _parse_proposals(self, response: str, num_proposals: int) -> List[str]:
        """
        Parse proposals from model response.
        Handles various formats (numbered lists, bullets, etc.)
        """
        lines = response.split('\n')
        proposals = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and headers
            if not line or line in ["Next steps:", "Proposals:", "possible next steps"]:
                continue
            
            # Remove common prefixes (1., -, •, etc.)
            for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '•', '*']:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            
            # Remove bracketed numbers like [1]
            if line and line[0].isdigit() and']' in line:
                line = line[line.index(']')+1:].strip()
            
            if line and len(line) > 3:  # Minimum reasonable length
                proposals.append(line)
            
            if len(proposals) >= num_proposals:
                break
        
        # If we couldn't parse enough, return what we have
        return proposals[:num_proposals]
    
    # ===================== EVALUATE FUNCTION =====================
    
    async def evaluate_state(
        self,
        problem: str,
        trajectory: str,
        llm_server: Dict,
    ) -> float:
        """
        Evaluate the quality/progress of current state.
        
        Args:
            problem: Original problem
            trajectory: Current solution trajectory
            llm_server: vLLM server config
            
        Returns:
            Value score between 0.0 and 1.0
        """
        # Check cache
        cache_key = f"evaluate_{hash(problem)}_{hash(trajectory)}"
        if self.config.cache_evaluations and cache_key in self.evaluation_cache:
            self.search_stats["cache_hits"] += 1
            return self.evaluation_cache[cache_key]
        
        self.search_stats["evaluations_performed"] += 1
        
        # Build evaluation prompt
        eval_prompt = self._build_evaluation_prompt(problem, trajectory)
        
        # Call model
        eval_response = await self._call_llm_streaming(llm_server, eval_prompt)
        
        # Parse evaluation into score
        score = self._parse_evaluation(eval_response)
        
        # Log evaluation
        eval_log = {
            "type": "state_evaluation",
            "timestamp": time.time(),
            "trajectory": trajectory,
            "prompt_preview": eval_prompt[:200] + "...",
            "response_preview": eval_response[:200] + "...",
            "score": score,
        }
        if self.decision_tree:
            if "evaluations" not in self.decision_tree[-1]:
                self.decision_tree[-1]["evaluations"] = []
            self.decision_tree[-1]["evaluations"].append(eval_log)
        
        # Cache
        if self.config.cache_evaluations:
            self.evaluation_cache[cache_key] = score
        
        return score
    
    def _build_evaluation_prompt(self, problem: str, trajectory: str) -> str:
        """Build dataset-aware evaluation prompts reused by ToT scoring."""
        return build_tot_value_prompt(problem, trajectory)
    
    def _parse_evaluation(self, response: str) -> float:
        """
        Parse evaluation response into a scalar score [0, 1]
        """
        response_lower = response.lower()
        
        confidence_keywords = {
            "sure": 0.9, "certain": 0.9, "confident": 0.9,
            "likely": 0.7, "probably": 0.7,
            "possible": 0.5, "maybe": 0.5,
            "unlikely": 0.3, "doubtful": 0.3,
            "impossible": 0.1, "blocked": 0.1,
        }
        
        for keyword, score in confidence_keywords.items():
            if keyword in response_lower:
                return score
        
        # Try to extract numeric score if present (1-9 scale)
        for i, char in enumerate(response):
            if char.isdigit():
                digit = int(char)
                if 1 <= digit <= 9:
                    return digit / 9.0  # Normalize to [0, 1]
        
        return 0.5  # Default neutral score
    
    # ===================== SEARCH IMPLEMENTATION =====================
    
    async def search(
        self,
        problem: str,
        llm_server: Dict,
    ) -> Dict[str, Any]:
        """
        Perform Tree of Thought search on the problem.
        
        Args:
            problem: Problem statement
            llm_server: vLLM server config
            
        Returns:
            Dictionary with best_trajectory, best_value, search_log
        """
        logger.info(f"Starting ToT search with method={self.config.search_method.value}")
        
        # Initialize root node
        self.root = TreeNode(trajectory="", depth=0, value=0.5)
        
        if self.config.search_method == SearchMethod.BFS:
            return await self._bfs_search(problem, llm_server)
        elif self.config.search_method == SearchMethod.BEAM:
            return await self._beam_search(problem, llm_server)
        else:
            return await self._dfs_search(problem, llm_server)
    
    async def _bfs_search(self, problem: str, llm_server: Dict) -> Dict[str, Any]:
        """Breadth-First Search implementation"""
        queue = [self.root]
        best_terminal = None
        best_value = 0.0
        best_candidate = None
        best_candidate_value = float('-inf')
        
        for depth in range(self.config.max_depth):
            if not queue:
                break
            
            next_queue = []
            
            for node in queue:
                # Generate proposals
                proposals = await self.propose_next_steps(
                    problem,
                    node.trajectory,
                    llm_server,
                    self.config.branching_factor
                )
                node.proposals = proposals
                
                # Create child nodes
                for prop in proposals:
                    new_trajectory = f"{node.trajectory}\n{prop}" if node.trajectory else prop
                    child = TreeNode(
                        trajectory=new_trajectory,
                        depth=depth + 1,
                        parent=node,
                    )
                    
                    # Evaluate
                    value = await self.evaluate_state(problem, new_trajectory, llm_server)
                    child.value = value
                    
                    # Track best candidate regardless of terminal status
                    if value > best_candidate_value:
                        best_candidate = child
                        best_candidate_value = value

                    # Check if terminal and meets threshold
                    if self._is_terminal(new_trajectory):
                        child.is_terminal = True
                        self.search_stats["solutions_found"] += 1
                        if value > best_value:
                            best_value = value
                            best_terminal = child
                        
                        # Early termination if high confidence
                        if self.config.early_termination and value >= self.config.sure_threshold:
                            return self._format_search_result(best_terminal, problem)
                    
                    # Prune low-value nodes
                    if value < self.config.impossible_threshold:
                        self.search_stats["branches_pruned"] += 1
                        continue
                    
                    node.children.append(child)
                    next_queue.append(child)
                    self.search_stats["total_nodes_in_tree"] += 1
            
            queue = next_queue[:self.config.max_candidates_per_level]
        
        return self._format_search_result(best_terminal or best_candidate, problem)
    
    async def _beam_search(self, problem: str, llm_server: Dict) -> Dict[str, Any]:
        """Beam Search implementation"""
        beam = [self.root]
        best_terminal = None
        best_value = 0.0
        best_candidate = None
        best_candidate_value = float('-inf')
        
        for depth in range(self.config.max_depth):
            candidates = []
            
            for node in beam:
                # Generate and evaluate proposals
                proposals = await self.propose_next_steps(
                    problem,
                    node.trajectory,
                    llm_server,
                    self.config.branching_factor
                )
                node.proposals = proposals
                
                for prop in proposals:
                    new_trajectory = f"{node.trajectory}\n{prop}" if node.trajectory else prop
                    value = await self.evaluate_state(problem, new_trajectory, llm_server)
                    
                    child = TreeNode(
                        trajectory=new_trajectory,
                        depth=depth + 1,
                        value=value,
                        parent=node,
                    )
                    
                    candidates.append((child, value))

                    if value > best_candidate_value:
                        best_candidate = child
                        best_candidate_value = value
                    self.search_stats["total_nodes_in_tree"] += 1
                    
                    if self._is_terminal(new_trajectory):
                        child.is_terminal = True
                        self.search_stats["solutions_found"] += 1
                        if value > best_value:
                            best_value = value
                            best_terminal = child
                        
                        if self.config.early_termination and value >= self.config.sure_threshold:
                            return self._format_search_result(best_terminal, problem)
            
            # Keep top-k by value
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = [child for child, _ in candidates[:self.config.beam_width]]
            
            if not beam:
                break
        
        return self._format_search_result(best_terminal or best_candidate, problem)
    
    async def _dfs_search(self, problem: str, llm_server: Dict) -> Dict[str, Any]:
        """Depth-First Search implementation"""
        best_terminal = None
        best_value = 0.0
        best_candidate = None
        best_candidate_value = float('-inf')
        
        async def dfs(node: TreeNode, depth: int):
            nonlocal best_terminal, best_value
            
            if depth >= self.config.max_depth:
                return
            
            # Generate proposals
            proposals = await self.propose_next_steps(
                problem,
                node.trajectory,
                llm_server,
                self.config.branching_factor
            )
            node.proposals = proposals
            
            for prop in proposals:
                new_trajectory = f"{node.trajectory}\n{prop}" if node.trajectory else prop
                value = await self.evaluate_state(problem, new_trajectory, llm_server)
                
                child = TreeNode(
                    trajectory=new_trajectory,
                    depth=depth + 1,
                    value=value,
                    parent=node,
                )
                node.children.append(child)

                if value > best_candidate_value:
                    best_candidate = child
                    best_candidate_value = value
                self.search_stats["total_nodes_in_tree"] += 1
                
                if self._is_terminal(new_trajectory):
                    child.is_terminal = True
                    self.search_stats["solutions_found"] += 1
                    if value > best_value:
                        best_value = value
                        best_terminal = child
                    
                    if self.config.early_termination and value >= self.config.sure_threshold:
                        return
                
                # Prune
                if value >= self.config.impossible_threshold:
                    await dfs(child, depth + 1)
                else:
                    self.search_stats["branches_pruned"] += 1
        
        await dfs(self.root, 0)
        return self._format_search_result(best_terminal or best_candidate, problem)
    
    # ===================== UTILITIES =====================
    
    def _is_terminal(self, trajectory: str) -> bool:
        """Check if trajectory represents a complete solution"""
        keywords = [
            "final answer",
            "reached goal",
            "solution:",
            "answer:",
            "conclusion:",
            "result:",
        ]
        trajectory_lower = trajectory.lower()
        return any(kw in trajectory_lower for kw in keywords)
    
    def _format_search_result(
        self,
        best_node: Optional[TreeNode],
        problem: str
    ) -> Dict[str, Any]:
        """Format search results for return"""
        if best_node:
            best_trajectory = best_node.trajectory
            best_value = best_node.value
        else:
            best_trajectory = ""
            best_value = 0.0
        
        return {
            "best_trajectory": best_trajectory,
            "best_value": best_value,
            "search_stats": self.search_stats,
            "decision_tree": self.decision_tree,
            "root_node": self.root,
        }
    
    async def _call_llm_streaming(
        self,
        llm_server: Dict,
        prompt: str
    ) -> str:
        """Call chat-completions endpoint and return the full response text."""
        import httpx

        payload = llm_server["payload"].copy()
        payload.pop("prompt", None)
        payload.pop("messages", None)
        payload["messages"] = [{"role": "user", "content": prompt}]
        payload["stream"] = False

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(
                    llm_server["url"],
                    headers=llm_server["headers"],
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise

        choices = data.get("choices", [])
        if not choices:
            logger.warning("LLM response missing choices: %s", data.keys())
            return ""

        choice = choices[0]
        if isinstance(choice, dict):
            msg = choice.get("message") or {}
            return msg.get("content") or choice.get("text", "")
        return str(choice)
    
    def get_decision_tree_json(self) -> str:
        """Export decision tree as JSON"""
        return json.dumps({
            "search_stats": self.search_stats,
            "decision_points": self.decision_tree,
            "num_decision_points": len(self.decision_tree),
        }, indent=2, default=str)
