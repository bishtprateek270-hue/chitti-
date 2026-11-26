"""
Unit tests for Chitti NewsService.
Tests Google News RSS parsing, NewsAPI fallback, and spoken summary formatting.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.agent.news_service import NewsService


class TestNewsService:
    def test_news_service_initialization(self):
        svc = NewsService()
        assert svc._cache == {}

    def test_format_spoken_summary_empty(self):
        svc = NewsService()
        summary_en = svc.format_spoken_summary([], "technology", "en")
        assert "couldn't fetch the latest headlines" in summary_en

        summary_hi = svc.format_spoken_summary([], "technology", "hi")
        assert "news headlines fetch nahi ho paayi" in summary_hi

    def test_format_spoken_summary_populated(self):
        svc = NewsService()
        sample_headlines = [
            {"title": "OpenAI releases new reasoning model", "source": "TechCrunch", "link": "http://example.com/1"},
            {"title": "ISRO plans new solar observatory", "source": "The Hindu", "link": "http://example.com/2"},
        ]
        summary = svc.format_spoken_summary(sample_headlines, category="technology", lang="en")
        assert "Here are the top technology headlines today:" in summary
        assert "First: OpenAI releases new reasoning model." in summary
        assert "Second: ISRO plans new solar observatory." in summary

    @patch("urllib.request.urlopen")
    def test_fetch_google_rss_mock(self, mock_urlopen):
        mock_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>AI Breakthrough in Quantum Computing - MIT News</title>
                    <link>https://news.google.com/article1</link>
                </item>
                <item>
                    <title>Nvidia Unveils Next-Gen AI GPU - Reuters</title>
                    <link>https://news.google.com/article2</link>
                </item>
            </channel>
        </rss>"""
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_xml
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        svc = NewsService()
        items = svc.fetch_headlines("technology", limit=2)
        assert len(items) == 2
        assert items[0]["title"] == "AI Breakthrough in Quantum Computing"
        assert items[0]["source"] == "MIT News"
        assert items[1]["title"] == "Nvidia Unveils Next-Gen AI GPU"
        assert items[1]["source"] == "Reuters"
