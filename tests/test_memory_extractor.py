"""Tests for Memory Extractor and Safety Filters (Phase 4A)."""

from src.memory.extractor import MemoryExtractor


def test_extract_explicit_remember_command():
    extractor = MemoryExtractor()

    cmd1 = extractor.extract_command("Remember that my main AI project is DocForensics AI.")
    assert cmd1.action == "remember"
    assert "DocForensics AI" in cmd1.content
    assert cmd1.importance >= 4

    cmd2 = extractor.extract_command("Chitti, please remember my favorite programming language is Python.")
    assert cmd2.action == "remember"
    assert "Python" in cmd2.content
    assert cmd2.memory_type == "preference"


def test_extract_compound_multi_fact_command():
    extractor = MemoryExtractor()

    cmd = extractor.extract_command("My name is Prateek Singh Bisht and I created you, also I'm an AIML engineer. Remember this.")
    assert cmd.action == "remember"
    assert len(cmd.facts) >= 3

    keys = [f.key for f in cmd.facts]
    assert "user_name" in keys
    assert "creator" in keys
    assert "occupation" in keys

    # Verify user name was resolved into creator
    creator_fact = next(f for f in cmd.facts if f.key == "creator")
    assert "Prateek Singh Bisht" in creator_fact.content


def test_extract_hindi_hinglish_memory_commands():
    extractor = MemoryExtractor()

    cmd1 = extractor.extract_command("Mera naam Prateek Singh Bisht hai.")
    assert cmd1.action == "remember"
    name_fact = next((f for f in cmd1.facts if f.key == "user_name"), None)
    assert name_fact is not None
    assert "Prateek" in name_fact.value

    cmd2 = extractor.extract_command("Maine tumhe banaya hai.")
    assert cmd2.action == "remember"
    creator_fact = next((f for f in cmd2.facts if f.key == "creator"), None)
    assert creator_fact is not None

    cmd3 = extractor.extract_command("Main AIML engineer hoon.")
    assert cmd3.action == "remember"
    occ_fact = next((f for f in cmd3.facts if f.key == "occupation"), None)
    assert occ_fact is not None
    assert "AIML" in occ_fact.value


def test_extract_forget_command():
    extractor = MemoryExtractor()

    cmd = extractor.extract_command("Forget that my favorite programming language is Python.")
    assert cmd.action == "forget"
    assert "Python" in cmd.content


def test_extract_forget_all_command():
    extractor = MemoryExtractor()

    cmd1 = extractor.extract_command("Forget everything about me.")
    assert cmd1.action == "forget_all"

    cmd2 = extractor.extract_command("Clear all memories")
    assert cmd2.action == "forget_all"


def test_extract_recall_command():
    extractor = MemoryExtractor()

    cmd1 = extractor.extract_command("Show me what you remember")
    assert cmd1.action == "recall"

    cmd2 = extractor.extract_command("What do you remember about my projects?")
    assert cmd2.action == "recall"


def test_sensitive_information_rejected():
    extractor = MemoryExtractor()

    secret_inputs = [
        "Remember that my password is supersecret123",
        "Keep in mind my API key is sk-1234567890abcdef1234567890abcdef",
        "Don't forget my access_token: secret_token_xyz_9988",
    ]

    for secret in secret_inputs:
        cmd = extractor.extract_command(secret)
        assert cmd.is_sensitive is True
        assert cmd.action == "sensitive_rejected"
        assert cmd.rejection_reason is not None


def test_casual_speech_not_an_explicit_command():
    extractor = MemoryExtractor()

    cmd = extractor.extract_command("What is machine learning?")
    assert cmd.action == "none"

    cmd2 = extractor.extract_command("How is the weather today?")
    assert cmd2.action == "none"
