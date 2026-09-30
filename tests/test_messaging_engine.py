"""
Tests for Chitti Messaging Engine & Dispatchers (WhatsApp & Gmail).
Verifies:
1. Native SMTP dispatch logic with mock and credentials.
2. Pre-filled Gmail Compose URL generation.
3. WhatsApp phone normalization and direct URL generation.
4. Tool engine execution for send_email and send_message.
5. Multi-step agent planning for email and WhatsApp messaging without regressions.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.agent.actions import ActionType
from src.agent.intent import ActionIntentAnalyzer, ActionIntentType
from src.agent.messaging import DispatchResult, EmailDispatcher, WhatsAppDispatcher
from src.agent.planner import AgentPlanner
from src.agent.projects import ProjectRegistry
from src.agent.tools import ToolEngine


class TestMessagingEngine(unittest.TestCase):
    """Test suite for WhatsApp and Gmail dispatchers."""

    def test_gmail_compose_url_generation(self):
        url = EmailDispatcher.get_gmail_compose_url(
            recipient="test@example.com",
            subject="Project Update",
            body="Hello team, here is the report.",
        )
        self.assertIn("https://mail.google.com/mail/u/0/?view=cm", url)
        self.assertIn("to=test%40example.com", url)
        self.assertIn("su=Project%20Update", url)
        self.assertIn("body=Hello%20team", url)

    def test_whatsapp_phone_detection_and_normalization(self):
        self.assertTrue(WhatsAppDispatcher.is_phone_number("+91 98765 43210"))
        self.assertTrue(WhatsAppDispatcher.is_phone_number("9876543210"))
        self.assertFalse(WhatsAppDispatcher.is_phone_number("Ayush Sharma"))
        self.assertFalse(WhatsAppDispatcher.is_phone_number("Rahul"))

        norm1 = WhatsAppDispatcher.normalize_phone_number("+91-9876543210")
        self.assertEqual(norm1, "919876543210")

        norm2 = WhatsAppDispatcher.normalize_phone_number("9876543210", default_country_code="91")
        self.assertEqual(norm2, "919876543210")

    def test_whatsapp_url_generation(self):
        url_contact = WhatsAppDispatcher.get_whatsapp_url("Ayush", "Hello")
        self.assertEqual(url_contact, "https://web.whatsapp.com")

        url_phone = WhatsAppDispatcher.get_whatsapp_url("+919876543210", "Meeting at 5pm")
        self.assertIn("https://web.whatsapp.com/send?phone=919876543210", url_phone)
        self.assertIn("text=Meeting%20at%205pm", url_phone)

    @patch("smtplib.SMTP")
    def test_email_smtp_dispatch_success(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = mock_server

        res = EmailDispatcher.send_email_smtp(
            recipient="user@example.com",
            subject="Status Update",
            body="All systems nominal.",
            sender_email="chitti@example.com",
            sender_password="secretpassword",
        )
        self.assertTrue(res.success)
        self.assertIn("successfully dispatched", res.message)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_with("chitti@example.com", "secretpassword")
        mock_server.send_message.assert_called_once()

    def test_email_smtp_no_credentials_fallback(self):
        # Without credentials, it should report NO_CREDENTIALS safely
        with patch.dict("os.environ", {}, clear=True):
            res = EmailDispatcher.send_email_smtp(
                recipient="user@example.com",
                subject="Status",
                body="Hello",
            )
            self.assertFalse(res.success)
            self.assertEqual(res.evidence, "NO_CREDENTIALS")

    def test_agent_intent_whatsapp_phone_and_name(self):
        intent_name = ActionIntentAnalyzer.extract_intent("send hi to ayush on whatsapp")
        self.assertEqual(intent_name.intent, ActionIntentType.SEND_MESSAGE)
        self.assertEqual(intent_name.target, "ayush")
        self.assertEqual(intent_name.content, "hi")

        intent_phone = ActionIntentAnalyzer.extract_intent("send message to 9876543210 saying meeting is confirmed on whatsapp")
        self.assertEqual(intent_phone.intent, ActionIntentType.SEND_MESSAGE)
        self.assertEqual(intent_phone.content, "meeting is confirmed")

    def test_agent_intent_email_extraction(self):
        intent = ActionIntentAnalyzer.extract_intent("send email to boss@company.com saying project is ready")
        self.assertEqual(intent.intent, ActionIntentType.SEND_EMAIL)
        self.assertEqual(intent.recipient, "boss@company.com")
        self.assertEqual(intent.content, "project is ready")

    def test_planner_creates_prefilled_email_plan(self):
        projects = MagicMock(spec=ProjectRegistry)
        planner = AgentPlanner(project_registry=projects)
        state = planner.plan_task("send email to dev@example.com saying code reviewed")
        self.assertIsNotNone(state)
        self.assertEqual(len(state.steps), 7)
        self.assertEqual(state.steps[0].action_type, "OPEN_URL")
        self.assertIn("mail.google.com", state.steps[0].parameters["url"])
        self.assertIn("dev%40example.com", state.steps[0].parameters["url"])

    def test_planner_creates_whatsapp_plan(self):
        projects = MagicMock(spec=ProjectRegistry)
        planner = AgentPlanner(project_registry=projects)
        state = planner.plan_task("send hi to ayush on whatsapp")
        self.assertIsNotNone(state)
        self.assertEqual(len(state.steps), 8)
        self.assertEqual(state.steps[0].action_type, "OPEN_URL")
        self.assertEqual(state.steps[3].action_type, "SEARCH_CONTACT")
        self.assertEqual(state.steps[6].action_type, "SEND_MESSAGE")


if __name__ == "__main__":
    unittest.main()
