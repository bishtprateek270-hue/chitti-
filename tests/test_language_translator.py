"""
Unit tests for Chitti Language Translator (Phase 4).
Tests dedicated translation mode, technical term preservation, and fallback translation.
"""

import pytest
from unittest.mock import MagicMock
from src.language.translator import Translator
from src.language.language_models import TranslationResult


def test_translator_with_mock_llm():
    mock_llm = MagicMock()
    mock_llm.generate_response.return_value = "मेरी कल एक आवश्यक बैठक है।"

    translator = Translator(llm=mock_llm)
    res = translator.translate(
        text="I have an important meeting tomorrow.",
        target_language="Hindi",
        source_language="en",
    )

    assert isinstance(res, TranslationResult)
    assert res.target_language == "hi"
    assert "बैठक" in res.translated_text
    assert mock_llm.generate_response.called


def test_translator_preserves_technical_terms():
    mock_llm = MagicMock()
    mock_llm.generate_response.return_value = "Machine learning models ko train karne ke liye GPU aur Python chahiye."

    translator = Translator(llm=mock_llm)
    res = translator.translate(
        text="To train machine learning models you need a GPU and Python.",
        target_language="Hinglish",
    )

    assert "Python" in res.preserved_terms or "GPU" in res.preserved_terms or "Machine Learning" in res.preserved_terms
    assert res.target_language == "hinglish"


def test_translator_fallback_translation():
    translator = Translator(llm=None)
    res = translator.translate(
        text="I have a meeting tomorrow.",
        target_language="Hindi",
    )
    assert res.target_language == "hi"
    assert len(res.translated_text) > 0


def test_translator_empty_text():
    translator = Translator(llm=None)
    res = translator.translate("", "Hindi")
    assert res.translated_text == ""
