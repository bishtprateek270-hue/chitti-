"""Tests for Chitti Conversation History and Personality Prompt."""

from src.brain.personality import ConversationHistory, CHITTI_SYSTEM_PROMPT


def test_initial_history_state():
    history = ConversationHistory()
    assert history.is_empty
    assert history.message_count == 0
    messages = history.get_messages_for_llm()
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == CHITTI_SYSTEM_PROMPT


def test_multi_turn_dialogue_accumulation():
    history = ConversationHistory()
    history.add_user_message("My name is Prateek.")
    history.add_assistant_message("Nice to meet you, Prateek!")
    history.add_user_message("What is my name?")

    assert history.message_count == 3
    assert not history.is_empty

    llm_msgs = history.get_messages_for_llm()
    assert len(llm_msgs) == 4  # system + 3 turns
    assert llm_msgs[1] == {"role": "user", "content": "My name is Prateek."}
    assert llm_msgs[2] == {"role": "assistant", "content": "Nice to meet you, Prateek!"}
    assert llm_msgs[3] == {"role": "user", "content": "What is my name?"}


def test_history_trimming():
    history = ConversationHistory(max_messages=4)
    for i in range(6):
        history.add_user_message(f"Message {i}")

    # Should keep only the last 4 messages
    assert history.message_count == 4
    llm_msgs = history.get_messages_for_llm()
    assert llm_msgs[1]["content"] == "Message 2"
    assert llm_msgs[-1]["content"] == "Message 5"


def test_clear_history():
    history = ConversationHistory()
    history.add_user_message("Hello")
    history.add_assistant_message("Hi")
    assert history.message_count == 2
    history.clear()
    assert history.is_empty
    assert history.message_count == 0
    assert len(history.get_messages_for_llm()) == 1  # Only system prompt
