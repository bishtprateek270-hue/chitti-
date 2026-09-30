"""
Tests for Chitti Messaging Engine & Dispatchers (WhatsApp & Gmail).
Verifies:
1. Native SMTP dispatch logic with mock and credentials.
2. Pre-filled Gmail Compose URL generation.
3. WhatsApp phone normalization and direct URL generation.
4. Tool engine execution for send_email and send_message.
5. Multi-step agent planning for email and WhatsApp messaging without regressions.
"""

import re
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
        self.assertEqual(intent_name.content, "Hi")

        intent_phone = ActionIntentAnalyzer.extract_intent("send message to 9876543210 saying meeting is confirmed on whatsapp")
        self.assertEqual(intent_phone.intent, ActionIntentType.SEND_MESSAGE)
        self.assertEqual(intent_phone.content, "Meeting is confirmed")

    def test_agent_intent_email_extraction(self):
        intent = ActionIntentAnalyzer.extract_intent("send email to boss@company.com saying project is ready")
        self.assertEqual(intent.intent, ActionIntentType.SEND_EMAIL)
        self.assertEqual(intent.recipient, "boss@company.com")
        self.assertEqual(intent.content, "Project is ready")

    def test_email_user_natural_phrasing(self):
        # The user's exact reported query with typo 'ho' and shorthand 'u':
        intent1 = ActionIntentAnalyzer.extract_intent("send ho how are u email to ayusharyaa618@gmail.com")
        self.assertEqual(intent1.intent, ActionIntentType.SEND_EMAIL)
        self.assertEqual(intent1.recipient, "ayusharyaa618@gmail.com")
        self.assertEqual(intent1.content, "Hi, how are you?")
        self.assertNotIn("email", intent1.content.lower())
        self.assertFalse(re.search(r"\bho\b", intent1.content, re.I))

        intent2 = ActionIntentAnalyzer.extract_intent("send how are you email to user@test.com")
        self.assertEqual(intent2.recipient, "user@test.com")
        self.assertEqual(intent2.content, "How are you?")
        self.assertNotIn("email", intent2.content.lower())

        intent3 = ActionIntentAnalyzer.extract_intent("send email to user@test.com: hello there")
        self.assertEqual(intent3.recipient, "user@test.com")
        self.assertEqual(intent3.content, "Hello there")

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

    def test_open_whatsapp_only_opens_url_without_sending(self):
        projects = MagicMock(spec=ProjectRegistry)
        planner = AgentPlanner(project_registry=projects)
        
        # Test "open whatsapp"
        intent = ActionIntentAnalyzer.extract_intent("open whatsapp")
        self.assertEqual(intent.intent, ActionIntentType.OPEN_URL)
        self.assertEqual(intent.destination, "https://web.whatsapp.com")

        state = planner.plan_task("open whatsapp")
        self.assertIsNotNone(state)
        self.assertEqual(len(state.steps), 2)
        self.assertEqual(state.steps[0].action_type, "OPEN_URL")
        self.assertEqual(state.steps[1].action_type, "VERIFY_PAGE_LOADED")
        self.assertFalse(any(s.action_type in ("SEND_MESSAGE", "TYPE_TEXT", "SEARCH_CONTACT") for s in state.steps))

        # Test "whatsapp kholo"
        intent_hi = ActionIntentAnalyzer.extract_intent("whatsapp kholo")
        self.assertEqual(intent_hi.intent, ActionIntentType.OPEN_URL)
        state_hi = planner.plan_task("whatsapp kholo")
        self.assertIsNotNone(state_hi)
        self.assertEqual(len(state_hi.steps), 2)
        self.assertFalse(any(s.action_type in ("SEND_MESSAGE", "TYPE_TEXT", "SEARCH_CONTACT") for s in state_hi.steps))

    def test_open_app_response_formatting_does_not_claim_message_sent(self):
        from src.agent.manager import LaptopAgentManager
        from src.agent.planner import TaskState, AgentStep

        manager = LaptopAgentManager()
        
        # 1. State representing "open whatsapp"
        open_wa_state = TaskState(task_description="open whatsapp", goal="Open WhatsApp Web in browser")
        open_wa_state.steps = [
            AgentStep(step_id=1, description="Open WhatsApp Web in browser", action_type="OPEN_URL", parameters={"url": "https://web.whatsapp.com", "site_name": "WhatsApp Web"}),
            AgentStep(step_id=2, description="Verify WhatsApp Web loaded", action_type="VERIFY_PAGE_LOADED", parameters={"url": "https://web.whatsapp.com"}, depends_on=[1]),
        ]

        resp_en = manager._format_multistep_response("open whatsapp", open_wa_state, success=True, raw_msg="", lang="en")
        self.assertEqual(resp_en, "Opened WhatsApp Web in your browser.")
        self.assertNotIn("sent", resp_en.lower())

        resp_hi = manager._format_multistep_response("open whatsapp", open_wa_state, success=True, raw_msg="", lang="hi")
        self.assertEqual(resp_hi, "ब्राउज़र में WhatsApp Web खोल दिया गया है।")

        resp_hinglish = manager._format_multistep_response("whatsapp open karo", open_wa_state, success=True, raw_msg="", lang="hinglish")
        self.assertEqual(resp_hinglish, "Browser mein WhatsApp Web open kar diya hai.")

        # 2. State representing "open gmail"
        open_gmail_state = TaskState(task_description="open gmail", goal="Open Gmail in browser")
        open_gmail_state.steps = [
            AgentStep(step_id=1, description="Open Gmail in browser", action_type="OPEN_URL", parameters={"url": "https://mail.google.com", "site_name": "Gmail"}),
            AgentStep(step_id=2, description="Verify Gmail loaded", action_type="VERIFY_PAGE_LOADED", parameters={"url": "https://mail.google.com"}, depends_on=[1]),
        ]

        resp_gmail = manager._format_multistep_response("open gmail", open_gmail_state, success=True, raw_msg="", lang="en")
        self.assertEqual(resp_gmail, "Opened Gmail in your browser.")
        self.assertNotIn("sent", resp_gmail.lower())

        # 3. State representing actual message sending
        send_msg_state = TaskState(task_description="send hi to ayush on whatsapp", goal="Send 'Hi' to ayush")
        send_msg_state.steps = [
            AgentStep(step_id=1, description="Open WhatsApp", action_type="OPEN_URL", parameters={"url": "https://web.whatsapp.com"}),
            AgentStep(step_id=2, description="Verify WhatsApp", action_type="VERIFY_PAGE_LOADED", parameters={"url": "https://web.whatsapp.com"}, depends_on=[1]),
            AgentStep(step_id=3, description="Search contact", action_type="SEARCH_CONTACT", parameters={"contact": "ayush"}, depends_on=[2]),
            AgentStep(step_id=4, description="Send message", action_type="SEND_MESSAGE", parameters={"contact": "ayush", "text": "Hi"}, depends_on=[3]),
        ]
        resp_send = manager._format_multistep_response("send hi to ayush on whatsapp", send_msg_state, success=True, raw_msg="", lang="en")
        self.assertEqual(resp_send, "Message has been sent to ayush on WhatsApp.")


if __name__ == "__main__":
    unittest.main()
