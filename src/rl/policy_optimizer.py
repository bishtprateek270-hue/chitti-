"""
Chitti RL Policy Optimizer & In-Context Policy Learning Engine.
Retrieves historical lessons and injects dynamic policy priors into the planner,
router, and conversation prompts so Chitti automatically avoids previous mistakes.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.rl.experience_replay import ExperienceReplayBuffer
from src.rl.models import Episode, ReflexionLesson, TaskOutcome, TrajectoryStep
from src.rl.reflexion_engine import ReflexionEngine
from src.rl.reward_evaluator import RewardEvaluator
from src.utils.logging import log_chitti, log_debug, log_info


class PolicyOptimizer:
    """
    Coordinates In-Context Policy Learning and Episodic Trajectory Logging.
    """

    def __init__(
        self,
        buffer: Optional[ExperienceReplayBuffer] = None,
        reflexion_engine: Optional[ReflexionEngine] = None,
    ):
        self.buffer = buffer or ExperienceReplayBuffer()
        self.reflexion = reflexion_engine or ReflexionEngine(buffer=self.buffer)
        self.current_episode: Optional[Episode] = None

    def start_episode(self, task_description: str, domain: str = "general") -> Episode:
        """Starts tracking a new execution episode."""
        ep_id = f"ep_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.current_episode = Episode(
            episode_id=ep_id,
            task_description=task_description,
            domain=domain,
        )
        log_debug(f"[RL POLICY] Started episode {ep_id} for task: '{task_description}'")
        return self.current_episode

    def record_step(
        self,
        action_type: str,
        action_payload: Dict[str, Any],
        observation_before: str = "",
        observation_after: str = "",
        verified: bool = True,
        error_message: Optional[str] = None,
        is_recovery: bool = False,
        execution_time_sec: float = 0.0,
    ) -> TrajectoryStep:
        """Records a step transition in the active episode and calculates its reward."""
        if self.current_episode is None:
            self.start_episode(task_description=f"Direct action: {action_type}")

        reward = RewardEvaluator.evaluate_step(
            verified=verified,
            error_message=error_message,
            is_recovery=is_recovery,
            execution_time_sec=execution_time_sec,
        )

        step = TrajectoryStep(
            step_id=len(self.current_episode.steps) + 1,
            action_type=action_type,
            action_payload=action_payload,
            observation_before=observation_before,
            observation_after=observation_after,
            reward=reward,
            verified=verified,
            error_message=error_message,
        )

        self.current_episode.steps.append(step)

        # Trigger immediate self-reflexion if step failed
        if reward < 0:
            lesson = self.reflexion.reflect_on_failure(
                task_description=self.current_episode.task_description,
                failed_step=step,
                domain=self.current_episode.domain,
                action_history=self.current_episode.steps,
            )
            self.current_episode.lessons_learned.append(lesson)

        return step

    def finish_episode(
        self,
        outcome: TaskOutcome = TaskOutcome.SUCCESS,
        user_feedback: Optional[float] = None,
    ) -> Episode:
        """Concludes the active episode, calculates cumulative reward, and commits to replay memory."""
        if self.current_episode is None:
            return None

        self.current_episode.outcome = outcome
        self.current_episode.user_feedback_score = user_feedback
        self.current_episode.end_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.current_episode.total_reward = RewardEvaluator.compute_episode_reward(
            steps=self.current_episode.steps,
            outcome=outcome,
            user_feedback=user_feedback,
        )

        self.buffer.record_episode(self.current_episode)
        ep = self.current_episode
        self.current_episode = None
        log_chitti(f"[RL POLICY] 📊 Episode finished: Outcome={outcome.value}, Total Reward={ep.total_reward:+.2f}")
        return ep

    def get_policy_guidance(self, task_query: str, domain: Optional[str] = None) -> str:
        """
        Retrieves relevant historical lessons learned from past mistakes
        and formats them as explicit policy guidelines for the LLM / Planner.
        """
        lessons = self.buffer.get_relevant_lessons(task_query, domain=domain, limit=3)
        if not lessons:
            return ""

        guidelines = [f"- {l.task_pattern}: {l.correction_rule}" for l in lessons]
        block = (
            "\n[RL POLICY: SELF-CORRECTION GUIDELINES LEARNED FROM PAST MISTAKES]\n"
            + "\n".join(guidelines)
            + "\n"
        )
        return block
