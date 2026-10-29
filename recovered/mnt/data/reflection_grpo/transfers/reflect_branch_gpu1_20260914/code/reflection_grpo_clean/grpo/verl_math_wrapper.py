"""
Wrapper for verl's built-in math scoring function.

This wrapper makes verl's compute_score compatible with the custom_reward_function
interface that expects additional parameters like data_source and extra_info.
"""

from typing import Optional, Union
from verl.utils.reward_score.math import compute_score as verl_compute_score


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Union[str, dict],
    extra_info: Optional[dict] = None,
) -> float:
    """
    Wrapper function for verl's math compute_score.
    
    This function accepts the standard custom_reward_function signature
    and delegates to verl's built-in math scoring logic.
    
    Args:
        data_source: Data source identifier (e.g., "math500") - not used
        solution_str: Model's generated solution
        ground_truth: Reference answer (str or dict with 'answer' key)
        extra_info: Optional extra information - not used
        
    Returns:
        Reward value (1.0 for correct, 0.0 for incorrect)
    """
    # Extract answer from ground_truth if it's a dict
    if isinstance(ground_truth, dict):
        gold_answer = ground_truth.get("answer", "")
    else:
        gold_answer = ground_truth
    
    # Call verl's built-in compute_score with the correct signature
    return verl_compute_score(solution_str, gold_answer)
