"""
Unit tests for Chitti GitHubService.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.agent.github_service import GitHubService


class TestGitHubService:
    def test_github_service_without_token(self):
        gh = GitHubService(token="")
        res = gh._make_request("/user")
        assert res["success"] is False
        assert "GITHUB_TOKEN not configured" in res["error"]

    @patch("urllib.request.urlopen")
    def test_get_authenticated_user_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = b'{"login":"testuser","name":"Test User","public_repos":12,"followers":5}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        gh = GitHubService(token="mock_token")
        user = gh.get_authenticated_user()
        assert user["success"] is True
        assert user["login"] == "testuser"
        assert user["name"] == "Test User"
        assert user["public_repos"] == 12

    @patch("urllib.request.urlopen")
    def test_list_repositories_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = b'[{"name":"repo1","full_name":"user/repo1","private":false,"stargazers_count":10}]'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        gh = GitHubService(token="mock_token")
        repos = gh.list_repositories()
        assert len(repos) == 1
        assert repos[0]["name"] == "repo1"
        assert repos[0]["stars"] == 10
