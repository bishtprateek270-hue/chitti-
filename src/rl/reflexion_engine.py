"""
Chitti Actor-Critic Reflexion & Self-Correction Engine.
Analyzes failed trajectories, diagnoses root causes, generates actionable
policy constraints, and produces DPO preference pairs so Chitti never repeats a mistake.
"""

import json
import re
from typing import Any, Dict, List, Optional

from src.brain.llm import BaseLLM
from src.rl.experience_replay import ExperienceReplayBuffer
from src.rl.models import DPOPreferencePair, Episode, ReflexionLesson, TaskOutcome, TrajectoryStep
from src.utils.logging import log_chitti, log_debug, log_info, log_warn


REFLEXION_PROMPT_TEMPLATE = """You are Chitti's Autonomous Reinforcement Learning Critic & Reflexion Engine.
An action or task execution just failed with a negative reward. Your goal is to perform a deep Root Cause Analysis and formulate a permanent, concise Self-Correction Guideline so that Chitti NEVER repeats this mistake.

[TASK DESCRIPTION]: {task_description}
[FAILED ACTION]: {failed_action}
[OBSERVATION / ERROR]: {error_message}
[ACTION SEQUENCE BEFORE FAILURE]: {action_history}

Please output ONLY a JSON object with this exact schema:
{{
    "root_cause": "Specific explanation of what assumption, parameter, or timing failed",
    "correction_rule": "Concrete actionable constraint (e.g. 'When opening X, wait Y ms before Z' or 'Verify button presence using OCR before clicking')",
    "domain": "{domain}"
}}
"""


class ReflexionEngine:
    """
    Diagnoses task execution errors and synthesizes persistent policy rules.
    """

    def __init__(
        self,
        llm: Optional[LLMEngine] = None,
        buffer: Optional[ExperienceReplayBuffer] = None,
    ):
        self.llm = llm
        self.buffer = buffer or ExperienceReplayBuffer()

    def reflect_on_failure(
        self,
        task_description: str,
        failed_step: TrajectoryStep,
        domain: str = "general",
        action_history: Optional[List[TrajectoryStep]] = None,
    ) -> ReflexionLesson:
        """
        Synthesizes a self-correction lesson from a failed action.
        """
        history_str = ""
        if action_history:
            history_str = " -> ".join([f"Step {s.step_id}: {s.action_type}" for s in action_history[-4:]])

        prompt = REFLEXION_PROMPT_TEMPLATE.format(
            task_description=task_description,
            failed_action=f"{failed_step.action_type}({json.dumps(failed_step.action_payload)})",
            error_message=failed_step.error_message or "Action verification failed.",
            action_history=history_str or "First step",
            domain=domain,
        )

        root_cause = "Action did not achieve expected verified state."
        correction_rule = f"Before executing {failed_step.action_type}, verify preconditions and add explicit synchronization delay."

        if self.llm:
            try:
                if hasattr(self.llm, "generate_response"):
                    raw_response = self.llm.generate_response([{"role": "user", "content": prompt}])
                elif hasattr(self.llm, "generate"):
                    raw_response = self.llm.generate(prompt)
                else:
                    raw_response = ""
                # Extract JSON block
                json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    root_cause = parsed.get("root_cause", root_cause)
                    correction_rule = parsed.get("correction_rule", correction_rule)
                    domain = parsed.get("domain", domain)
            except Exception as e:
                log_debug(f"[REFLEXION] LLM reflexion synthesis notice: {e}")

        lesson = ReflexionLesson(
            task_pattern=task_description,
            domain=domain,
            failed_action=f"{failed_step.action_type}",
            root_cause=root_cause,
            correction_rule=correction_rule,
            reward_delta=-1.0,
        )

        self.buffer.store_lesson(lesson)
        log_chitti(f"[RL REFLEXION] 🧠 Learned self-correction lesson for '{task_description}': {correction_rule}")
        return lesson

    def create_dpo_pair(
        self,
        prompt: str,
        rejected_plan: str,
        chosen_plan: str,
        domain: str = "general",
    ) -> DPOPreferencePair:
        """
        Constructs and records a Direct Preference Optimization (DPO) sample from failure & recovery.
        """
        pair = DPOPreferencePair(
            prompt=prompt,
            chosen_response=chosen_plan,
            rejected_response=rejected_plan,
            domain=domain,
        )
        self.buffer.store_dpo_pair(pair)
        log_debug(f"[RL DPO] Recorded preference pair for prompt: '{prompt}'")
        return pair
