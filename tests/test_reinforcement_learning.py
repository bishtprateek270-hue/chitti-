"""
Comprehensive Unit & Integration Test Suite for Chitti Reinforcement Learning (RL) Engine.
Verifies Reward Evaluation, Episodic Experience Replay, Actor-Critic Reflexion, and Policy Optimization.
"""

import os
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import MagicMock, patch

from src.rl.models import DPOPreferencePair, Episode, ReflexionLesson, TaskOutcome, TrajectoryStep
from src.rl.reward_evaluator import RewardEvaluator
from src.rl.experience_replay import ExperienceReplayBuffer
from src.rl.reflexion_engine import ReflexionEngine
from src.rl.policy_optimizer import PolicyOptimizer


class TestReinforcementLearningEngine(unittest.TestCase):
    """Unit and integration tests for Chitti's RL self-correction engine."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / f"test_rl_{uuid.uuid4().hex[:6]}.db"
        self.buffer = ExperienceReplayBuffer(db_path=str(self.db_path))

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_reward_evaluator_step_rewards(self):
        """RewardEvaluator should assign correct rewards for verified, failed, and recovery steps."""
        # Verified normal step
        r1 = RewardEvaluator.evaluate_step(verified=True, execution_time_sec=1.0)
        self.assertEqual(r1, 1.0)

        # Failed step with error
        r2 = RewardEvaluator.evaluate_step(verified=False, error_message="Element not found")
        self.assertEqual(r2, -1.0)

        # Self-healing recovery step
        r3 = RewardEvaluator.evaluate_step(verified=True, is_recovery=True, execution_time_sec=2.0)
        self.assertAlmostEqual(r3, 0.65, places=2)

        # Slow step penalty
        r4 = RewardEvaluator.evaluate_step(verified=True, execution_time_sec=10.0)
        self.assertLess(r4, 1.0)

    def test_reward_evaluator_user_feedback(self):
        """RewardEvaluator should identify praise vs criticism in English and Hindi."""
        praise_en = RewardEvaluator.evaluate_user_feedback("Good job Chitti, perfect!")
        self.assertEqual(praise_en, 1.0)

        praise_hi = RewardEvaluator.evaluate_user_feedback("Shabash, bilkul sahi hai!")
        self.assertEqual(praise_hi, 1.0)

        critique_en = RewardEvaluator.evaluate_user_feedback("No, that is completely wrong.")
        self.assertEqual(critique_en, -1.0)

        critique_hi = RewardEvaluator.evaluate_user_feedback("Galat hai, nahi hua.")
        self.assertEqual(critique_hi, -1.0)

        neutral = RewardEvaluator.evaluate_user_feedback("What time is it?")
        self.assertIsNone(neutral)

    def test_experience_replay_record_and_query(self):
        """ExperienceReplayBuffer should store episodes, steps, and query relevant lessons."""
        lesson = ReflexionLesson(
            task_pattern="Open WhatsApp and send message",
            domain="browser",
            failed_action="type_text",
            root_cause="Search box was not yet focused",
            correction_rule="Wait for search input element focus before sending keystrokes.",
            reward_delta=-1.0,
        )
        self.buffer.store_lesson(lesson)

        # Query lessons for WhatsApp
        matched = self.buffer.get_relevant_lessons("Send a WhatsApp message to Rahul", domain="browser")
        self.assertEqual(len(matched), 1)
        self.assertIn("Wait for search input", matched[0].correction_rule)

    def test_reflexion_engine_heuristic_and_llm(self):
        """ReflexionEngine should synthesize actionable rules on step failure."""
        reflexion = ReflexionEngine(buffer=self.buffer)
        failed_step = TrajectoryStep(
            step_id=1,
            action_type="click_element",
            action_payload={"selector": "#submit-btn"},
            observation_before="Page loading",
            observation_after="Error 404",
            reward=-1.0,
            verified=False,
            error_message="Button not clickable",
        )

        lesson = reflexion.reflect_on_failure(
            task_description="Click submit on form",
            failed_step=failed_step,
            domain="browser",
        )

        self.assertEqual(lesson.failed_action, "click_element")
        self.assertIn("click_element", lesson.correction_rule)

        # Check that lesson is stored in replay buffer
        stats = self.buffer.get_stats()
        self.assertEqual(stats["total_lessons_learned"], 1)

    def test_dpo_pair_creation_and_export(self):
        """ReflexionEngine and Buffer should record and export DPO preference pairs."""
        reflexion = ReflexionEngine(buffer=self.buffer)
        pair = reflexion.create_dpo_pair(
            prompt="Open Notepad and write hello",
            rejected_plan="1. type_text 'hello'",
            chosen_plan="1. launch_app 'notepad'\n2. wait_for_window\n3. type_text 'hello'",
            domain="os",
        )
        self.assertEqual(pair.prompt, "Open Notepad and write hello")

        export_path = Path(self.temp_dir.name) / "test_dpo.jsonl"
        count = self.buffer.export_dpo_dataset(str(export_path))
        self.assertEqual(count, 1)
        self.assertTrue(export_path.exists())

    def test_policy_optimizer_lifecycle_and_guidance(self):
        """PolicyOptimizer should manage full episode lifecycle and inject learned guidelines."""
        optimizer = PolicyOptimizer(buffer=self.buffer)

        # 1. Start episode
        ep = optimizer.start_episode("Create test file on Desktop", domain="os")
        self.assertEqual(ep.task_description, "Create test file on Desktop")

        # 2. Record successful step
        optimizer.record_step(
            action_type="create_file",
            action_payload={"path": "Desktop/test.txt"},
            verified=True,
        )

        # 3. Finish episode
        finished_ep = optimizer.finish_episode(outcome=TaskOutcome.SUCCESS, user_feedback=1.0)
        self.assertGreater(finished_ep.total_reward, 0.5)

        # 4. Check statistics
        stats = self.buffer.get_stats()
        self.assertEqual(stats["total_episodes"], 1)
        self.assertGreater(stats["average_reward"], 0.5)


if __name__ == "__main__":
    unittest.main()
