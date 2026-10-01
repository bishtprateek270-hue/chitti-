"""
Chitti RL Reward Evaluator.
Computes fine-grained reward signals based on environment verification,
execution latency, self-healing recovery attempts, and natural language user feedback.
"""

import re
from typing import Dict, Any, List, Optional
from src.rl.models import TaskOutcome, TrajectoryStep


class RewardEvaluator:
    """
    Evaluates actions and full episodes to assign dense and sparse reward values.
    Standardized on [-1.0, +1.0] scale.
    """

    POSITIVE_FEEDBACK_REGEX = re.compile(
        r"(?i)\b(?:good\s+job|great|perfect|awesome|well\s+done|correct|nice|sahi\s+hai|shabash|badhiya|bilkul\s+sahi|thank\s+you|thanks)\b"
    )
    NEGATIVE_FEEDBACK_REGEX = re.compile(
        r"(?i)\b(?:wrong|incorrect|mistake|failed|bad|not\s+working|galat\s+hai|nahi\s+hua|gadbad|error|stop|that'?s\s+not\s+what\s+i\s+asked)\b"
    )

    @classmethod
    def evaluate_step(
        cls,
        verified: bool,
        error_message: Optional[str] = None,
        is_recovery: bool = False,
        execution_time_sec: float = 0.0,
    ) -> float:
        """
        Calculates immediate reward for a single state-action step:
        - Verified success: +0.8 to +1.0
        - Action failed: -0.8 to -1.0
        - Recovery action: -0.3 penalty
        """
        if not verified or error_message:
            return -1.0

        base_reward = 1.0
        if is_recovery:
            base_reward -= 0.35

        # Small latency penalty for unnecessarily slow steps (> 5s)
        if execution_time_sec > 5.0:
            latency_penalty = min(0.2, (execution_time_sec - 5.0) * 0.02)
            base_reward -= latency_penalty

        return max(-1.0, min(1.0, base_reward))

    @classmethod
    def evaluate_user_feedback(cls, user_text: str) -> Optional[float]:
        """
        Extracts implicit or explicit reinforcement reward from user's follow-up reply.
        Returns +1.0 for praise, -1.0 for negative correction, or None if neutral.
        """
        if not user_text or not user_text.strip():
            return None

        clean = user_text.strip()
        if cls.NEGATIVE_FEEDBACK_REGEX.search(clean):
            return -1.0
        if cls.POSITIVE_FEEDBACK_REGEX.search(clean):
            return 1.0
        return None

    @classmethod
    def compute_episode_reward(
        cls,
        steps: List[TrajectoryStep],
        outcome: TaskOutcome,
        user_feedback: Optional[float] = None,
    ) -> float:
        """
        Computes discounted cumulative episode reward incorporating step history and user feedback.
        """
        if not steps:
            return -1.0 if outcome == TaskOutcome.FAILED else 0.0

        step_rewards = [s.reward for s in steps]
        gamma = 0.95
        discounted_sum = sum((gamma ** i) * r for i, r in enumerate(step_rewards))
        avg_step_reward = discounted_sum / len(steps)

        # Sparse outcome reward
        outcome_bonus = 1.0 if outcome == TaskOutcome.SUCCESS else (-1.0 if outcome == TaskOutcome.FAILED else 0.2)

        total = 0.4 * avg_step_reward + 0.4 * outcome_bonus
        if user_feedback is not None:
            total = 0.5 * total + 0.5 * user_feedback

        return round(float(max(-1.0, min(1.0, total))), 3)
