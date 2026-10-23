"""
Chitti Application, Resource & Folder Dynamic Discovery Registry.
Dynamically resolves installed applications, user folders, and web resources
without rigid hardcoded command lists.
"""

import os
import sys
import re
import shutil
import platform
import difflib
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Any

try:
    import winreg
except ImportError:
    winreg = None

from src.utils.logging import log_debug, log_info, log_warning


@dataclass
class DiscoveredApp:
    name: str
    path: Optional[str]
    executable: str
    is_running: bool = False


# Generic descriptive semantic aliases
SEMANTIC_APP_ALIASES: Dict[str, str] = {
    "editor": "code",
    "my editor": "code",
    "the editor": "code",
    "code editor": "code",
    "my code editor": "code",
    "browser": "chrome",
    "the browser": "chrome",
    "my browser": "chrome",
    "the web browser": "chrome",
    "web browser": "chrome",
    "that browser": "chrome",
    "terminal": "wt",
    "the terminal": "wt",
    "my terminal": "wt",
    "command line": "cmd",
    "the command line": "cmd",
    "music player": "spotify",
    "the music player": "spotify",
    "music site": "youtube",
    "the music site": "youtube",
    "music app": "spotify",
    "the music app": "spotify",
    "messaging app": "whatsapp",
    "the messaging app": "whatsapp",
    "chat app": "whatsapp",
    "the chat app": "whatsapp",
    "mail": "gmail",
    "the mail": "gmail",
    "my mail": "gmail",
    "email": "gmail",
    "the email": "gmail",
    "email app": "gmail",
    "calculator": "calc",
    "the calculator": "calc",
    "task manager": "taskmgr",
    "the task manager": "taskmgr",
    "file manager": "explorer",
    "the file manager": "explorer",
    "explorer": "explorer",
    "the explorer": "explorer",
    "paint": "mspaint",
    "paint app": "mspaint",
    "the paint": "mspaint",
}

# Backward compatibility alias map
DEFAULT_APP_MAP: Dict[str, List[str]] = {
    "chrome": ["chrome.exe", "google chrome", "chrome"],
    "google chrome": ["chrome.exe", "google chrome"],
    "vs code": ["code.cmd", "code.exe", "code"],
    "vscode": ["code.cmd", "code.exe", "code"],
    "code": ["code.cmd", "code.exe", "code"],
    "visual studio code": ["code.cmd", "code.exe", "code"],
    "notepad": ["notepad.exe", "notepad"],
    "calculator": ["calc.exe", "calculator"],
    "calc": ["calc.exe", "calculator"],
    "terminal": ["wt.exe", "cmd.exe", "powershell.exe"],
    "windows terminal": ["wt.exe"],
    "cmd": ["cmd.exe"],
    "command prompt": ["cmd.exe"],
    "powershell": ["powershell.exe"],
    "file explorer": ["explorer.exe"],
    "explorer": ["explorer.exe"],
    "paint": ["mspaint.exe", "paint"],
    "mspaint": ["mspaint.exe"],
    "task manager": ["taskmgr.exe"],
    "taskmgr": ["taskmgr.exe"],
    "spotify": ["spotify.exe", "spotify"],
    "word": ["winword.exe"],
    "excel": ["excel.exe"],
}

# Standard URL mappings for popular services
DEFAULT_URL_MAP: Dict[str, str] = {
    # Search & AI
    "google": "https://www.google.com",
    "gemini": "https://gemini.google.com",
    "chatgpt": "https://chatgpt.com",
    "claude": "https://claude.ai",
    "deepseek": "https://www.deepseek.com",
    "perplexity": "https://www.perplexity.ai",
    "openai": "https://openai.com",
    "bing": "https://www.bing.com",
    "yahoo": "https://www.yahoo.com",
    "duckduckgo": "https://duckduckgo.com",

    # Social, Streaming & Media
    "youtube": "https://www.youtube.com",
    "yt": "https://www.youtube.com",
    "twitter": "https://www.x.com",
    "x": "https://www.x.com",
    "x.com": "https://www.x.com",
    "instagram": "https://www.instagram.com",
    "facebook": "https://www.facebook.com",
    "reddit": "https://www.reddit.com",
    "linkedin": "https://www.linkedin.com",
    "spotify": "https://open.spotify.com",
    "netflix": "https://www.netflix.com",
    "prime video": "https://www.primevideo.com",
    "hotstar": "https://www.hotstar.com",
    "jiocinema": "https://www.jiocinema.com",
    "twitch": "https://www.twitch.tv",
    "pinterest": "https://www.pinterest.com",
    "imdb": "https://www.imdb.com",

    # Messaging & Productivity
    "whatsapp": "https://web.whatsapp.com",
    "whatsapp web": "https://web.whatsapp.com",
    "telegram": "https://web.telegram.org",
    "discord": "https://discord.com/app",
    "slack": "https://app.slack.com",
    "teams": "https://teams.microsoft.com",
    "gmail": "https://mail.google.com",
    "outlook": "https://outlook.live.com",
    "notion": "https://www.notion.so",
    "figma": "https://www.figma.com",
    "canva": "https://www.canva.com",

    # Developer, Knowledge & Coding Platforms
    "github": "https://www.github.com",
    "gitlab": "https://gitlab.com",
    "bitbucket": "https://bitbucket.org",
    "stackoverflow": "https://stackoverflow.com",
    "wikipedia": "https://www.wikipedia.org",
    "quora": "https://www.quora.com",
    "medium": "https://medium.com",
    "substack": "https://substack.com",
    "leetcode": "https://leetcode.com",
    "geeksforgeeks": "https://www.geeksforgeeks.org",
    "gfg": "https://www.geeksforgeeks.org",
    "hackerrank": "https://www.hackerrank.com",
    "codechef": "https://www.codechef.com",
    "codeforces": "https://codeforces.com",
    "huggingface": "https://huggingface.co",
    "kaggle": "https://www.kaggle.com",
    "arxiv": "https://arxiv.org",
    "npm": "https://www.npmjs.com",
    "pypi": "https://pypi.org",
    "vercel": "https://vercel.com",
    "netlify": "https://www.netlify.com",

    # E-Commerce, Food, Cricket & Lifestyle
    "amazon": "https://www.amazon.com",
    "amazon india": "https://www.amazon.in",
    "amazon in": "https://www.amazon.in",
    "amazon.in": "https://www.amazon.in",
    "flipkart": "https://www.flipkart.com",
    "myntra": "https://www.myntra.com",
    "swiggy": "https://www.swiggy.com",
    "zomato": "https://www.zomato.com",
    "cricbuzz": "https://www.cricbuzz.com",
    "espn": "https://www.espn.com",
    "espncricinfo": "https://www.espncricinfo.com",
    "bookmyshow": "https://in.bookmyshow.com",

    # Learning & Courses
    "coursera": "https://www.coursera.org",
    "udemy": "https://www.udemy.com",
    "khan academy": "https://www.khanacademy.org",

    # Generic Browser / Website Landing Targets
    "browser": "https://www.google.com",
    "the browser": "https://www.google.com",
    "my browser": "https://www.google.com",
    "website": "https://www.google.com",
    "the website": "https://www.google.com",
    "a website": "https://www.google.com",
    "web": "https://www.google.com",
    "homepage": "https://www.google.com",


    # User's Portfolio & GitHub Links
    "portfolio": "https://bishtprateek270-hue.github.io/Portfolio/",
    "my portfolio": "https://bishtprateek270-hue.github.io/Portfolio/",
    "portfolio website": "https://bishtprateek270-hue.github.io/Portfolio/",
    "portfolio site": "https://bishtprateek270-hue.github.io/Portfolio/",
    "portfolio link": "https://bishtprateek270-hue.github.io/Portfolio/",
    "my projects": "https://bishtprateek270-hue.github.io/Portfolio/",
    "projects": "https://bishtprateek270-hue.github.io/Portfolio/",
    "my github": "https://github.com/bishtprateek270-hue",
    "my github profile": "https://github.com/bishtprateek270-hue",
    "github profile": "https://github.com/bishtprateek270-hue",
    "my repos": "https://github.com/bishtprateek270-hue?tab=repositories",
    "my repo": "https://github.com/bishtprateek270-hue?tab=repositories",
    "my repositories": "https://github.com/bishtprateek270-hue?tab=repositories",
    "my repository": "https://github.com/bishtprateek270-hue?tab=repositories",
    "my github repo": "https://github.com/bishtprateek270-hue?tab=repositories",
    "my github repos": "https://github.com/bishtprateek270-hue?tab=repositories",
    "my github repository": "https://github.com/bishtprateek270-hue?tab=repositories",
    "my github repositories": "https://github.com/bishtprateek270-hue?tab=repositories",
    "github repo": "https://github.com/bishtprateek270-hue?tab=repositories",
    "github repos": "https://github.com/bishtprateek270-hue?tab=repositories",
    "github repository": "https://github.com/bishtprateek270-hue?tab=repositories",
    "github repositories": "https://github.com/bishtprateek270-hue?tab=repositories",
    "my git repo": "https://github.com/bishtprateek270-hue?tab=repositories",
    "git repo": "https://github.com/bishtprateek270-hue?tab=repositories",
    "repos": "https://github.com/bishtprateek270-hue?tab=repositories",
    "repo": "https://github.com/bishtprateek270-hue?tab=repositories",
    "repositories": "https://github.com/bishtprateek270-hue?tab=repositories",
    "repository": "https://github.com/bishtprateek270-hue?tab=repositories",
    "folder of github repo": "https://github.com/bishtprateek270-hue?tab=repositories",
    "the folder of github repo": "https://github.com/bishtprateek270-hue?tab=repositories",
    "folder of github repo in browser": "https://github.com/bishtprateek270-hue?tab=repositories",
    "the folder of github repo in browser": "https://github.com/bishtprateek270-hue?tab=repositories",
    "folder of repo": "https://github.com/bishtprateek270-hue?tab=repositories",
    "the folder of repo": "https://github.com/bishtprateek270-hue?tab=repositories",
    "github repo folder": "https://github.com/bishtprateek270-hue?tab=repositories",
    "github folder": "https://github.com/bishtprateek270-hue?tab=repositories",
    "repo folder": "https://github.com/bishtprateek270-hue?tab=repositories",
    "the repo folder": "https://github.com/bishtprateek270-hue?tab=repositories",
    "my linkedin": "https://linkedin.com/in/prateek-singh-bisht",

    # Generic Live Deployment Links
    "deployment": "https://3-d-ulpin-mvp.vercel.app/",
    "deployment link": "https://3-d-ulpin-mvp.vercel.app/",
    "the deployment link": "https://3-d-ulpin-mvp.vercel.app/",
    "live deployment": "https://3-d-ulpin-mvp.vercel.app/",
    "live deployment link": "https://3-d-ulpin-mvp.vercel.app/",
    "the live deployment link": "https://3-d-ulpin-mvp.vercel.app/",
    "deployment link of the project": "https://3-d-ulpin-mvp.vercel.app/",
    "deployment link of project": "https://3-d-ulpin-mvp.vercel.app/",
    "live deployment link of the project": "https://3-d-ulpin-mvp.vercel.app/",
    "live deployment link of project": "https://3-d-ulpin-mvp.vercel.app/",
    "the deployment link of the project": "https://3-d-ulpin-mvp.vercel.app/",
    "the live deployment link of the project": "https://3-d-ulpin-mvp.vercel.app/",
    "ldeployment link": "https://3-d-ulpin-mvp.vercel.app/",
    "ldeployment link of the project": "https://3-d-ulpin-mvp.vercel.app/",
    "the ldeployment link of the project": "https://3-d-ulpin-mvp.vercel.app/",
    "live link": "https://3-d-ulpin-mvp.vercel.app/",
    "live link of the project": "https://3-d-ulpin-mvp.vercel.app/",
    "live link of project": "https://3-d-ulpin-mvp.vercel.app/",
    "live demo": "https://3-d-ulpin-mvp.vercel.app/",
    "project deployment": "https://3-d-ulpin-mvp.vercel.app/",
    "project live link": "https://3-d-ulpin-mvp.vercel.app/",
    "project demo": "https://3-d-ulpin-mvp.vercel.app/",
    "project demo link": "https://3-d-ulpin-mvp.vercel.app/",

    # User's Projects (Demos & Repositories)
    "chitti": "https://github.com/bishtprateek270-hue/chitti-",
    "chitti repo": "https://github.com/bishtprateek270-hue/chitti-",
    "chitti repository": "https://github.com/bishtprateek270-hue/chitti-",
    "chitti github": "https://github.com/bishtprateek270-hue/chitti-",

    "3d-ulpin": "https://3-d-ulpin-mvp.vercel.app/",
    "3d ulpin": "https://3-d-ulpin-mvp.vercel.app/",
    "3d unplin": "https://3-d-ulpin-mvp.vercel.app/",
    "3d-unplin": "https://3-d-ulpin-mvp.vercel.app/",
    "3d usb": "https://3-d-ulpin-mvp.vercel.app/",
    "3d-ulpin demo": "https://3-d-ulpin-mvp.vercel.app/",
    "3d ulpin demo": "https://3-d-ulpin-mvp.vercel.app/",
    "3d unplin demo": "https://3-d-ulpin-mvp.vercel.app/",
    "3d usb demo": "https://3-d-ulpin-mvp.vercel.app/",
    "3d-ulpin website": "https://3-d-ulpin-mvp.vercel.app/",
    "3d ulpin website": "https://3-d-ulpin-mvp.vercel.app/",
    "3d unplin website": "https://3-d-ulpin-mvp.vercel.app/",
    "3d usb website": "https://3-d-ulpin-mvp.vercel.app/",
    "3d ulpin deployment": "https://3-d-ulpin-mvp.vercel.app/",
    "3d ulpin deployment link": "https://3-d-ulpin-mvp.vercel.app/",
    "3d ulpin live": "https://3-d-ulpin-mvp.vercel.app/",
    "3d ulpin live link": "https://3-d-ulpin-mvp.vercel.app/",
    "3d ulpin live deployment": "https://3-d-ulpin-mvp.vercel.app/",
    "3d ulpin live deployment link": "https://3-d-ulpin-mvp.vercel.app/",
    "3d ulpin ldeployment link": "https://3-d-ulpin-mvp.vercel.app/",
    "3d ulpin ldeployment": "https://3-d-ulpin-mvp.vercel.app/",
    "3d-ulpin deployment": "https://3-d-ulpin-mvp.vercel.app/",
    "3d-ulpin deployment link": "https://3-d-ulpin-mvp.vercel.app/",
    "3d-ulpin live": "https://3-d-ulpin-mvp.vercel.app/",
    "3d-ulpin live link": "https://3-d-ulpin-mvp.vercel.app/",
    "3d unplin deployment": "https://3-d-ulpin-mvp.vercel.app/",
    "3d unplin deployment link": "https://3-d-ulpin-mvp.vercel.app/",
    "3d-ulpin repo": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",
    "3d ulpin repo": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",
    "3d unplin repo": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",
    "3d-unplin repo": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",
    "3d usb repo": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",
    "3d-ulpin repository": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",
    "3d ulpin repository": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",
    "3d unplin repository": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",
    "3d usb repository": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",
    "3d-ulpin github": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",
    "3d ulpin github": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",
    "3d unplin github": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",
    "3d usb github": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",
    "3d ulpin github folder": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",
    "folder of github repo of 3d ulpin": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",
    "folder of github repo for 3d ulpin": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",
    "folder of 3d ulpin repo": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",
    "folder of 3d unplin repo": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",

    "karigarai": "https://karigar-ai-amber.vercel.app",
    "karigar ai": "https://karigar-ai-amber.vercel.app",
    "karigar": "https://karigar-ai-amber.vercel.app",
    "karigarai demo": "https://karigar-ai-amber.vercel.app",
    "karigar ai demo": "https://karigar-ai-amber.vercel.app",
    "karigar demo": "https://karigar-ai-amber.vercel.app",
    "karigarai website": "https://karigar-ai-amber.vercel.app",
    "karigar ai website": "https://karigar-ai-amber.vercel.app",
    "karigarai repo": "https://github.com/bishtprateek270-hue/karigarAI",
    "karigar ai repo": "https://github.com/bishtprateek270-hue/karigarAI",
    "karigar repo": "https://github.com/bishtprateek270-hue/karigarAI",
    "karigarai repository": "https://github.com/bishtprateek270-hue/karigarAI",
    "karigar ai repository": "https://github.com/bishtprateek270-hue/karigarAI",
    "karigar repository": "https://github.com/bishtprateek270-hue/karigarAI",
    "karigarai github": "https://github.com/bishtprateek270-hue/karigarAI",
    "karigar ai github": "https://github.com/bishtprateek270-hue/karigarAI",
    "karigar github": "https://github.com/bishtprateek270-hue/karigarAI",

    "docforensics": "https://doc-forensics-ai.vercel.app",
    "docforensics ai": "https://doc-forensics-ai.vercel.app",
    "doc forensics": "https://doc-forensics-ai.vercel.app",
    "doc forensics ai": "https://doc-forensics-ai.vercel.app",
    "docforensics demo": "https://doc-forensics-ai.vercel.app",
    "docforensics website": "https://doc-forensics-ai.vercel.app",
    "docforensics repo": "https://github.com/bishtprateek270-hue/DocForensics-AI",
    "docforensics ai repo": "https://github.com/bishtprateek270-hue/DocForensics-AI",
    "doc forensics repo": "https://github.com/bishtprateek270-hue/DocForensics-AI",
    "doc forensics ai repo": "https://github.com/bishtprateek270-hue/DocForensics-AI",
    "docforensics repository": "https://github.com/bishtprateek270-hue/DocForensics-AI",
    "docforensics github": "https://github.com/bishtprateek270-hue/DocForensics-AI",

    "doseband": "https://github.com/bishtprateek270-hue/DoseBand",
    "dose band": "https://github.com/bishtprateek270-hue/DoseBand",
    "doseband repo": "https://github.com/bishtprateek270-hue/DoseBand",
    "dose band repo": "https://github.com/bishtprateek270-hue/DoseBand",
    "doseband repository": "https://github.com/bishtprateek270-hue/DoseBand",
    "doseband github": "https://github.com/bishtprateek270-hue/DoseBand",

    "resume analyzer": "https://resume-analyzer-nejevwpy2bcvrvuag9schr.streamlit.app/",
    "ai resume analyzer": "https://resume-analyzer-nejevwpy2bcvrvuag9schr.streamlit.app/",
    "resume analyzer demo": "https://resume-analyzer-nejevwpy2bcvrvuag9schr.streamlit.app/",
    "resume analyzer website": "https://resume-analyzer-nejevwpy2bcvrvuag9schr.streamlit.app/",
    "resume analyzer repo": "https://github.com/bishtprateek270-hue/resume-analyzer",
    "ai resume analyzer repo": "https://github.com/bishtprateek270-hue/resume-analyzer",
    "resume analyzer repository": "https://github.com/bishtprateek270-hue/resume-analyzer",
    "resume analyzer github": "https://github.com/bishtprateek270-hue/resume-analyzer",

    "ragmind": "https://ai-rag-platform-sigma.vercel.app",
    "ai rag platform": "https://ai-rag-platform-sigma.vercel.app",
    "ai-rag-platform": "https://ai-rag-platform-sigma.vercel.app",
    "rag platform": "https://ai-rag-platform-sigma.vercel.app",
    "ragmind demo": "https://ai-rag-platform-sigma.vercel.app",
    "ragmind website": "https://ai-rag-platform-sigma.vercel.app",
    "rag repo": "https://github.com/bishtprateek270-hue/AI-RAG-Platform",
    "ragmind repo": "https://github.com/bishtprateek270-hue/AI-RAG-Platform",
    "ragmind repository": "https://github.com/bishtprateek270-hue/AI-RAG-Platform",
    "ragmind github": "https://github.com/bishtprateek270-hue/AI-RAG-Platform",

    "shikshasetu": "https://shikshasetu-ten.vercel.app",
    "shiksha setu": "https://shikshasetu-ten.vercel.app",
    "siksasetu": "https://shikshasetu-ten.vercel.app",
    "siksa setu": "https://shikshasetu-ten.vercel.app",
    "shikshasetu demo": "https://shikshasetu-ten.vercel.app",
    "shikshasetu website": "https://shikshasetu-ten.vercel.app",
    "shikshasetu repo": "https://github.com/bishtprateek270-hue/ShikshaSetu",
    "shiksha setu repo": "https://github.com/bishtprateek270-hue/ShikshaSetu",
    "siksasetu repo": "https://github.com/bishtprateek270-hue/ShikshaSetu",
    "siksa setu repo": "https://github.com/bishtprateek270-hue/ShikshaSetu",
    "shikshasetu repository": "https://github.com/bishtprateek270-hue/ShikshaSetu",
    "shiksha setu repository": "https://github.com/bishtprateek270-hue/ShikshaSetu",
    "siksasetu repository": "https://github.com/bishtprateek270-hue/ShikshaSetu",
    "shikshasetu github": "https://github.com/bishtprateek270-hue/ShikshaSetu",
    "message repository": "https://github.com/bishtprateek270-hue/ShikshaSetu",

    "churnops": "https://churnops-u0bn.onrender.com",
    "churn ops": "https://churnops-u0bn.onrender.com",
    "churnops demo": "https://churnops-u0bn.onrender.com",
    "churnops website": "https://churnops-u0bn.onrender.com",
    "churnops repo": "https://github.com/bishtprateek270-hue/churnops",
    "churn ops repo": "https://github.com/bishtprateek270-hue/churnops",
    "churnops repository": "https://github.com/bishtprateek270-hue/churnops",
    "churnops github": "https://github.com/bishtprateek270-hue/churnops",

    "simplepay": "https://simplepay-946w.onrender.com",
    "simple pay": "https://simplepay-946w.onrender.com",
    "simplepay demo": "https://simplepay-946w.onrender.com",
    "simplepay website": "https://simplepay-946w.onrender.com",
    "simplepay repo": "https://github.com/bishtprateek270-hue/SimplePay",
    "simple pay repo": "https://github.com/bishtprateek270-hue/SimplePay",
    "simplepay repository": "https://github.com/bishtprateek270-hue/SimplePay",
    "simplepay github": "https://github.com/bishtprateek270-hue/SimplePay",

    "puzzle app": "https://puzzle-app-53uh.onrender.com",
    "puzzle app demo": "https://puzzle-app-53uh.onrender.com",
    "puzzle app website": "https://puzzle-app-53uh.onrender.com",
    "puzzle app repo": "https://github.com/bishtprateek270-hue/puzzle-app",
    "puzzle app repository": "https://github.com/bishtprateek270-hue/puzzle-app",
    "puzzle app github": "https://github.com/bishtprateek270-hue/puzzle-app",

    "dsa repo": "https://github.com/bishtprateek270-hue/dsa",
    "dsa repository": "https://github.com/bishtprateek270-hue/dsa",
    "dsa github": "https://github.com/bishtprateek270-hue/dsa",

    "mlops repo": "https://github.com/bishtprateek270-hue/ml-ops",
    "ml ops repo": "https://github.com/bishtprateek270-hue/ml-ops",
    "mlops repository": "https://github.com/bishtprateek270-hue/ml-ops",
    "mlops github": "https://github.com/bishtprateek270-hue/ml-ops",
}


class AppDiscovery:
    """Dynamically discovers and resolves application executable paths and shortcuts on the host OS."""

    _cached_app_paths: Optional[Dict[str, str]] = None

    @classmethod
    def _normalize_name(cls, app_name: str) -> str:
        clean = app_name.strip().lower()
        if clean in SEMANTIC_APP_ALIASES:
            return SEMANTIC_APP_ALIASES[clean]
        
        # Check without determiners
        stripped = re.sub(r"^(the|my|that|this|a|an)\s+", "", clean).strip()
        if stripped in SEMANTIC_APP_ALIASES:
            return SEMANTIC_APP_ALIASES[stripped]
            
        return clean

    @classmethod
    def get_registered_app_paths(cls) -> Dict[str, str]:
        """Queries the Windows Registry App Paths for all registered machine and user applications."""
        if cls._cached_app_paths is not None:
            return cls._cached_app_paths

        app_paths: Dict[str, str] = {}
        if platform.system() != "Windows" or winreg is None:
            cls._cached_app_paths = app_paths
            return app_paths

        reg_roots = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"),
            (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\App Paths"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths"),
        ]

        for root_key, sub_key in reg_roots:
            try:
                with winreg.OpenKey(root_key, sub_key) as key:
                    num_subkeys, _, _ = winreg.QueryInfoKey(key)
                    for i in range(num_subkeys):
                        try:
                            app_sub = winreg.EnumKey(key, i)
                            with winreg.OpenKey(key, app_sub) as app_key:
                                exec_path, _ = winreg.QueryValueEx(app_key, "")
                                if exec_path:
                                    exec_path_clean = exec_path.strip('"')
                                    base_name = app_sub.lower().replace(".exe", "")
                                    app_paths[base_name] = exec_path_clean
                                    app_paths[app_sub.lower()] = exec_path_clean
                        except Exception:
                            continue
            except Exception:
                continue

        cls._cached_app_paths = app_paths
        return app_paths

    @classmethod
    def find_start_menu_shortcuts(cls, query: str) -> Optional[str]:
        """Searches Start Menu program shortcuts for matching applications."""
        if platform.system() != "Windows":
            return None

        clean_query = query.lower().replace(" ", "")
        program_data = os.environ.get("ProgramData", "C:\\ProgramData")
        app_data = os.environ.get("APPDATA", "")

        search_dirs = [
            Path(program_data) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
            Path(app_data) / "Microsoft" / "Windows" / "Start Menu" / "Programs" if app_data else None,
        ]

        for s_dir in search_dirs:
            if not s_dir or not s_dir.exists():
                continue
            try:
                for lnk in s_dir.rglob("*.lnk"):
                    stem_clean = lnk.stem.lower().replace(" ", "")
                    if clean_query == stem_clean or clean_query in stem_clean or stem_clean in clean_query:
                        return str(lnk.resolve())
            except Exception:
                continue

        return None

    @classmethod
    def resolve_application(cls, app_name: str) -> Optional[str]:
        """
        Dynamically resolves an application target to an executable path, shortcut, or shell command.
        Uses PATH lookup, Windows Registry App Paths, Start Menu shortcut discovery, and standard Program Files.
        """
        raw_clean = app_name.strip()
        clean = cls._normalize_name(raw_clean)
        clean_no_ext = clean.replace(".exe", "").replace(".cmd", "").replace(".bat", "")

        # 0. Check Windows built-in utility commands
        builtins = {
            "notepad": "notepad.exe",
            "calc": "calc.exe",
            "calculator": "calc.exe",
            "explorer": "explorer.exe",
            "file explorer": "explorer.exe",
            "cmd": "cmd.exe",
            "command prompt": "cmd.exe",
            "powershell": "powershell.exe",
            "terminal": "wt.exe",
            "windows terminal": "wt.exe",
            "wt": "wt.exe",
            "taskmgr": "taskmgr.exe",
            "task manager": "taskmgr.exe",
            "mspaint": "mspaint.exe",
            "paint": "mspaint.exe",
            "control": "control.exe",
            "settings": "ms-settings:",
        }
        if clean in builtins or clean_no_ext in builtins:
            b_target = builtins.get(clean, builtins.get(clean_no_ext))
            if shutil.which(b_target):
                return shutil.which(b_target)
            return b_target

        # 1. Check system PATH via shutil.which
        candidate_names = [clean, f"{clean}.exe", f"{clean}.cmd", f"{clean}.bat", clean_no_ext, f"{clean_no_ext}.exe"]
        if clean in ("vs code", "vscode", "visual studio code"):
            candidate_names = ["code.cmd", "code.exe", "code"] + candidate_names
        elif clean in ("chrome", "google chrome"):
            candidate_names = ["chrome.exe", "chrome"] + candidate_names

        for c in candidate_names:
            resolved = shutil.which(c)
            if resolved:
                return resolved

        # 2. Check Windows Registry App Paths
        reg_apps = cls.get_registered_app_paths()
        if clean in reg_apps:
            path_val = reg_apps[clean]
            if os.path.exists(path_val):
                return path_val
        if clean_no_ext in reg_apps:
            path_val = reg_apps[clean_no_ext]
            if os.path.exists(path_val):
                return path_val

        # Substring search in registry apps
        for reg_k, reg_v in reg_apps.items():
            if clean_no_ext == reg_k or clean_no_ext in reg_k or reg_k in clean_no_ext:
                if os.path.exists(reg_v):
                    return reg_v

        # 3. Check Start Menu Shortcuts (.lnk)
        start_menu_lnk = cls.find_start_menu_shortcuts(clean_no_ext)
        if start_menu_lnk:
            return start_menu_lnk

        # 4. Search Common Program Directories
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        user_profile = os.environ.get("USERPROFILE", "")

        search_roots = [
            Path(local_app_data) / "Programs" if local_app_data else None,
            Path(program_files),
            Path(program_files_x86),
            Path(local_app_data) if local_app_data else None,
            Path(user_profile) if user_profile else None,
        ]

        for root in search_roots:
            if not root or not root.exists():
                continue
            try:
                # Direct folder check
                app_dir = root / clean_no_ext.title()
                if app_dir.exists() and app_dir.is_dir():
                    for exe_candidate in [f"{clean_no_ext}.exe", f"{clean_no_ext.title()}.exe"]:
                        target_p = app_dir / exe_candidate
                        if target_p.exists():
                            return str(target_p.resolve())
            except Exception:
                continue

        # 5. Check if it's an executable file directly provided
        direct_p = Path(raw_clean)
        if direct_p.exists() and direct_p.is_file():
            return str(direct_p.resolve())

        # 6. Fuzzy Match against known apps, builtins, and registered apps
        all_known = list(builtins.keys()) + list(DEFAULT_APP_MAP.keys()) + list(reg_apps.keys())
        fuzzy_app = difflib.get_close_matches(clean_no_ext, all_known, n=1, cutoff=0.7)
        if fuzzy_app and fuzzy_app[0] != clean_no_ext:
            matched_name = fuzzy_app[0]
            log_debug(f"[APP_DISCOVERY] Fuzzy matched '{clean_no_ext}' -> '{matched_name}'")
            return cls.resolve_application(matched_name)

        # Do not treat known web services as local executables if no local executable exists
        if clean_no_ext in DEFAULT_URL_MAP and clean_no_ext not in DEFAULT_APP_MAP:
            return None

        # Fallback: if it's a simple alphanumeric name, allow standard Windows shell execution
        if re.match(r"^[a-zA-Z0-9_\-]+$", clean_no_ext):
            return clean_no_ext

        return None

    # Alias for convenience
    resolve_app = resolve_application


class FolderDiscovery:
    """Resolves standard user folder directories and aliases safely."""

    @classmethod
    def resolve_folder_path(cls, folder_alias: str) -> Optional[Path]:
        raw = folder_alias.strip() if folder_alias else ""
        clean = raw.lower().strip(".!? \t\n\"'")
        # Strip common leading verbs and determiners
        clean = re.sub(r"^(?:open\s+|show\s+|launch\s+|the\s+|my\s+|a\s+|an\s+|this\s+|that\s+|some\s+)+", "", clean).strip()
        # Strip trailing keywords
        clean = re.sub(r"(?:\s+(?:folder|directory|dir|kholo|dikhao|open\s+karo|karo))+$", "", clean).strip()

        home = Path.home()
        desktop = (home / "OneDrive" / "Desktop") if (home / "OneDrive" / "Desktop").exists() else (home / "Desktop")
        documents = (home / "OneDrive" / "Documents") if (home / "OneDrive" / "Documents").exists() else (home / "Documents")
        pictures = (home / "OneDrive" / "Pictures") if (home / "OneDrive" / "Pictures").exists() else (home / "Pictures")

        # Generic / empty folder request -> default to Desktop or Home
        if not clean or clean in (
            "folder", "directory", "dir", "folders", "directories", "files",
            "my files", "file explorer", "explorer", "my pc", "this pc", "computer",
            "a", "the", "my", "some"
        ):
            return desktop if desktop.exists() else home

        alias_map = {
            "downloads": home / "Downloads",
            "download": home / "Downloads",
            "documents": documents,
            "docs": documents,
            "doc": documents,
            "document": documents,
            "desktop": desktop,
            "pictures": pictures,
            "photos": pictures,
            "pics": pictures,
            "images": pictures,
            "picture": pictures,
            "videos": home / "Videos",
            "video": home / "Videos",
            "movies": home / "Videos",
            "music": home / "Music",
            "songs": home / "Music",
            "audio": home / "Music",
            "home": home,
            "user": home,
            "userprofile": home,
            "profile": home,
            "workspace": Path.cwd(),
            "work": Path.cwd(),
            "current": Path.cwd(),
            "cwd": Path.cwd(),
            "project": Path.cwd(),
            "chitti": Path.cwd(),
            "c drive": Path("C:\\"),
            "c:": Path("C:\\"),
            "c": Path("C:\\"),
        }

        # Check standard alias
        if clean in alias_map:
            p = alias_map[clean]
            if p.exists():
                return p

        # Check direct path
        try:
            for cand_str in (raw, clean):
                direct_path = Path(cand_str)
                if direct_path.is_absolute() and direct_path.exists() and direct_path.is_dir():
                    return direct_path

            # Check relative to standard locations
            search_bases = [Path.cwd(), desktop, home / "Downloads", documents, home]
            for base in search_bases:
                if not base or not base.exists():
                    continue
                cand_rel = base / clean
                if cand_rel.exists() and cand_rel.is_dir():
                    return cand_rel

            # Search immediate subdirectories (case-insensitive)
            for base in search_bases:
                if not base or not base.exists():
                    continue
                try:
                    for child in base.iterdir():
                        if child.is_dir() and child.name.lower() == clean:
                            return child
                except Exception:
                    continue
        except Exception:
            pass

        # Fallback to Desktop or Home rather than returning None if a folder was requested
        return desktop if desktop.exists() else home


class ResourceDiscovery:
    """Categorizes arbitrary user targets into URLs, Folders, Applications, or Files."""

    @classmethod
    def resolve_url(cls, target: str) -> Optional[str]:
        """
        Intelligently resolves arbitrary user phrases (including 'website X', 'open X on browser',
        'the website of X', 'X ka repo', 'project X', etc.) to a verified URL or valid web scheme.
        """
        if not target or not isinstance(target, str):
            return None
        from src.utils.text import normalize_typos
        raw = target.strip()
        normalized_raw = normalize_typos(raw)
        clean = normalized_raw.lower().strip()

        # 1. Direct web scheme or domain check
        if clean.startswith("http://") or clean.startswith("https://"):
            return raw
        if clean.startswith("www."):
            return f"https://{raw}"
        domain_pattern = r"^(?:https?://)?(?:[a-zA-Z0-9_\-]+\.)+[a-zA-Z]{2,}(?::\d+)?(?:/\S*)?$"
        if re.search(domain_pattern, clean):
            return f"https://{raw}" if not clean.startswith(("http://", "https://")) else raw

        # 2. Direct match in DEFAULT_URL_MAP
        if clean in DEFAULT_URL_MAP:
            return DEFAULT_URL_MAP[clean]

        # 3. Handle Hinglish / Hindi postpositions: e.g. "3d ulpin ka repo", "flipkart ki website", "cricbuzz ki site"
        m_hinglish = re.search(r"(?i)^([a-z0-9_\-\s]+?)\s+(?:ka|ki|ke)\s+(repo|repository|github|git|link|website|site|url|demo|live)\b", clean)
        if m_hinglish:
            h_subj = m_hinglish.group(1).strip()
            h_type = m_hinglish.group(2).strip().lower()
            if h_type in ("repo", "repository", "github", "git"):
                cand_key = f"{h_subj} repo"
            else:
                cand_key = f"{h_subj}"
            if cand_key in DEFAULT_URL_MAP:
                return DEFAULT_URL_MAP[cand_key]
            clean = f"{h_type} of {h_subj}"

        # 4. Check if request is asking for general user repos or profile
        generic_repo_queries = {
            "my repo", "repo", "my repos", "repos", "my repository", "repository",
            "my repositories", "repositories", "my github repo", "my github repos",
            "github repo", "github repos", "my github repository", "github repository",
            "my git repo", "git repo", "my git repos", "git repos",
            "my github", "github", "github profile", "my github profile",
            "folder of github repo", "the folder of github repo",
            "folder of github repo in browser", "the folder of github repo in browser",
            "folder of repo", "the folder of repo", "github repo folder", "github folder",
            "repo folder", "the repo folder",
        }
        if clean in generic_repo_queries:
            if clean in ("my github", "github", "github profile", "my github profile"):
                return "https://github.com/bishtprateek270-hue"
            return "https://github.com/bishtprateek270-hue?tab=repositories"

        # 4B. Check if request is asking for general deployment link
        generic_deploy_queries = {
            "deployment", "deployment link", "the deployment link",
            "live deployment", "live deployment link", "the live deployment link",
            "deployment link of the project", "deployment link of project",
            "live deployment link of the project", "live deployment link of project",
            "the deployment link of the project", "the live deployment link of the project",
            "ldeployment link", "ldeployment link of the project", "the ldeployment link of the project",
            "live link", "live link of the project", "live link of project",
            "live demo", "project deployment", "project live link", "project demo",
            "the deployment link of the projects that i have created",
            "deployment link of the projects that i have created",
            "live deployment link of the projects that i have created",
            "deployment link of the projects", "live deployment link of the projects",
        }
        if clean in generic_deploy_queries:
            return "https://3-d-ulpin-mvp.vercel.app/"

        # 4C. Check generic browser / website requests
        generic_web_queries = {
            "website", "the website", "a website", "websites", "the websites",
            "browser", "the browser", "my browser", "web", "the web",
            "internet", "homepage", "home page", "new tab", "browser tab",
            "open website", "open the website", "open browser",
        }
        if clean in generic_web_queries:
            return "https://www.google.com"

        # 5. Check if request is specifically asking for a REPO vs DEPLOYMENT/DEMO vs WEBSITE
        is_repo_request = bool(re.search(
            r"(?i)\b(?:folder\s+of\s+github\s+repo|folder\s+of\s+repo|github\s+repo\s+folder|repo\s+folder|github\s+folder|github\s+repo|github\s+repository|github|repo|repository|git\s+repo|source\s+code)\b",
            clean
        ))
        is_deployment_request = bool(re.search(
            r"(?i)\b(?:live\s+deployment\s+link|deployment\s+link|live\s+deployment|deployment|ldeployment\s+link|ldeployment|live\s+link|live\s+url|live\s+site|live\s+demo|live\s+app|production\s+link|demo\s+link|live)\b",
            clean
        ))
        is_website_request = bool(re.search(
            r"(?i)\b(?:website|site|webpage|web|portal|page|in\s+browser|on\s+browser|browser\s+me|browser\s+mein|browser\s+par|browser\s+pe)\b",
            clean
        ))

        # 6. Normalize target by stripping prefix noise:
        clean_norm = clean
        # Strip "browser me / on browser" prefixes
        clean_norm = re.sub(r"(?i)^(?:browser\s+(?:me|mein|par|pe)\s+|(?:in|on)\s+(?:the\s+|my\s+)?browser\s+)", "", clean_norm).strip()
        # Strip leading phrases like "website called X", "website of X", "folder of github repo of X"
        clean_norm = re.sub(r"(?i)^(?:the\s+|my\s+|a\s+|an\s+)?(?:folder\s+of\s+github\s+repo|folder\s+of\s+repo|github\s+repo\s+folder|repo\s+folder|github\s+folder|github\s+|git\s+)?(?:repo|repository|live\s+deployment\s+link|deployment\s+link|live\s+deployment|deployment|ldeployment\s+link|ldeployment|live\s+link|demo\s+link|website|site|webpage|page|url|project|demo)\s+(?:named|called|of|for|to|is|with\s+name|by\s+name)\s*", "", clean_norm).strip()
        clean_norm = re.sub(r"(?i)^(?:the\s+|my\s+|a\s+|an\s+)?(?:folder\s+of\s+github\s+repo|folder\s+of\s+repo|github\s+repo\s+folder|repo\s+folder|github\s+folder|github\s+|git\s+)?(?:repo|repository|live\s+deployment\s+link|deployment\s+link|live\s+deployment|deployment|ldeployment\s+link|ldeployment|website|site|webpage|page|web|portal|link|url)\s+", "", clean_norm).strip()
        clean_norm = re.sub(r"(?i)^(?:the\s+|my\s+|a\s+|an\s+)?(?:project)\s+(?:named|called|of|for|to)?\s*", "", clean_norm).strip()
        clean_norm = re.sub(r"(?i)^(?:the\s+|my\s+|a\s+|an\s+)", "", clean_norm).strip()
        clean_norm = clean_norm.strip("\"' ")

        # Check if clean_norm is already an explicit domain
        if re.search(domain_pattern, clean_norm):
            return f"https://{clean_norm}" if not clean_norm.startswith(("http://", "https://")) else clean_norm

        # Strip suffix noise: e.g. "3d-ulpin repo", "karigar ai github", "3d ulpin deployment link", "cricbuzz on browser", "flipkart website"
        core_name = re.sub(r"(?i)\s+(?:repo|repository|github|git|folder\s+of\s+github\s+repo|github\s+folder|live\s+deployment\s+link|deployment\s+link|live\s+deployment|deployment|ldeployment\s+link|ldeployment|link|live\s+link|demo\s+link|website|site|webpage|page|portal|url|project|demo|live|(?:in|on)\s+(?:any\s+|my\s+|the\s+)?(?:browser|chrome|edge)|browser\s+(?:me|mein|par|pe))$", "", clean_norm).strip().strip("\"' ")
        core_name = re.sub(r"(?i)^(?:of\s+the\s+projects?\s+that\s+i\s+(?:have\s+)?created|of\s+the\s+project|of\s+project|of\s+the\s+app|of\s+app|of\s+|for\s+)\s*", "", core_name).strip()

        # Check if core_name is an explicit domain
        if re.search(domain_pattern, core_name):
            return f"https://{core_name}" if not core_name.startswith(("http://", "https://")) else core_name

        # Check clean_norm and core_name for generic repo keywords after stripping
        if not core_name or core_name in generic_repo_queries or core_name in ("project", "the project", "my project", "app", "the app", "projects", "my projects"):
            if is_deployment_request:
                return "https://3-d-ulpin-mvp.vercel.app/"
            if is_repo_request:
                return "https://github.com/bishtprateek270-hue?tab=repositories"
            if is_website_request:
                return "https://www.google.com"

        # If deployment link is requested, prioritize demo/deployment keys
        if is_deployment_request:
            deploy_candidates = [
                f"{core_name} deployment link",
                f"{core_name} deployment",
                f"{core_name} live link",
                f"{core_name} live",
                f"{core_name} demo",
                f"{core_name} website",
                core_name,
                f"{core_name.replace('-', ' ')} demo",
                f"{core_name.replace(' ', '-')} demo",
            ]
            for cand in deploy_candidates:
                if cand and cand in DEFAULT_URL_MAP:
                    return DEFAULT_URL_MAP[cand]

        # If repo is specifically requested, prioritize repo keys
        if is_repo_request:
            repo_candidates = [
                f"{core_name} repo",
                f"{clean_norm} repo" if clean_norm != core_name else None,
                f"{core_name.replace('-', ' ')} repo",
                f"{core_name.replace(' ', '-')} repo",
                f"{core_name} repository",
                f"{core_name} github",
            ]
            for cand in repo_candidates:
                if cand and cand in DEFAULT_URL_MAP:
                    return DEFAULT_URL_MAP[cand]

        # Check if clean or core_name is known to be a desktop app (and not explicitly asking for repo/link/website)
        if not is_repo_request and not is_website_request and not bool(re.search(r"(?i)\b(?:link|website|site|url|demo|live)\b", clean)):
            if clean in DEFAULT_APP_MAP or core_name in DEFAULT_APP_MAP or clean in ("vscode", "vs code", "code", "notepad", "calculator", "calc", "cmd", "terminal", "powershell", "paint", "explorer"):
                return None

        # Check clean_norm and core_name directly in DEFAULT_URL_MAP
        for cand in [clean_norm, core_name, core_name.replace('-', ' '), core_name.replace(' ', '-')]:
            if cand in DEFAULT_URL_MAP:
                return DEFAULT_URL_MAP[cand]

        # If link / demo / site requested, or fallback
        for cand in [f"{core_name} website", f"{core_name} demo", f"{core_name} link"]:
            if cand in DEFAULT_URL_MAP:
                return DEFAULT_URL_MAP[cand]

        # Check any repo key for core_name regardless
        for cand in [f"{core_name} repo", f"{core_name} repository", f"{core_name} github"]:
            if cand in DEFAULT_URL_MAP:
                return DEFAULT_URL_MAP[cand]

        # 7. Fuzzy matching on keys (strict cutoff to avoid matching vscode to discord)
        all_keys = list(DEFAULT_URL_MAP.keys())
        if is_repo_request:
            m = difflib.get_close_matches(f"{core_name} repo", all_keys, n=1, cutoff=0.7)
            if m:
                return DEFAULT_URL_MAP[m[0]]
            m_core = difflib.get_close_matches(f"{core_name}", all_keys, n=1, cutoff=0.7)
            if m_core:
                if f"{m_core[0]} repo" in DEFAULT_URL_MAP:
                    return DEFAULT_URL_MAP[f"{m_core[0]} repo"]
                return DEFAULT_URL_MAP[m_core[0]]

        m = difflib.get_close_matches(core_name, all_keys, n=1, cutoff=0.82)
        if m:
            return DEFAULT_URL_MAP[m[0]]

        # 8. Dynamic Website Target Resolution:
        # If the user explicitly commanded to open a website or browser ("open website X", "open X on browser", "X website kholo")
        # and core_name is a single alphanumeric brand/service name (e.g. "cricbuzz", "nytimes", "flipkart"):
        if is_website_request and re.match(r"^[a-zA-Z0-9_\-]+$", core_name) and len(core_name) >= 2:
            return f"https://www.{core_name}.com"

        return None

    @classmethod
    def is_url(cls, target: str) -> bool:
        if not target or not isinstance(target, str):
            return False
        clean = target.strip().lower()
        if clean.startswith("http://") or clean.startswith("https://") or clean.startswith("www."):
            return True
        domain_pattern = r"^(?:https?://)?(?:[a-zA-Z0-9_\-]+\.)+[a-zA-Z]{2,}(?::\d+)?(?:/\S*)?$"
        if re.search(domain_pattern, clean):
            return True
        if cls.resolve_url(target) is not None:
            return True
        return False

    @classmethod
    def to_url(cls, target: str) -> str:
        resolved = cls.resolve_url(target)
        if resolved:
            return resolved
        clean = target.strip()
        if clean.lower().startswith("http://") or clean.lower().startswith("https://"):
            return clean
        if clean.lower().startswith("www."):
            return f"https://{clean}"
        domain_pattern = r"^(?:https?://)?(?:[a-zA-Z0-9_\-]+\.)+[a-zA-Z]{2,}(?::\d+)?(?:/\S*)?$"
        if re.search(domain_pattern, clean.lower()):
            return f"https://{clean}"

        # If asking for a repo/github, open repositories list or search GitHub
        if re.search(r"(?i)\b(?:repo|repository|github|git)\b", clean):
            core = re.sub(r"(?i)^(?:the\s+|my\s+|a\s+|an\s+)?(?:github\s+|git\s+)?(?:repo|repository)\s+(?:named|called|of|for|to)?\s*", "", clean).strip()
            if core and core != clean:
                return f"https://github.com/bishtprateek270-hue?tab=repositories&q={urllib.parse.quote_plus(core)}"
            return "https://github.com/bishtprateek270-hue?tab=repositories"

        # If single alphanumeric token without spaces, construct standard domain
        clean_lower = clean.lower()
        if re.match(r"^[a-zA-Z0-9_\-]+$", clean_lower) and len(clean_lower) >= 2:
            return f"https://www.{clean_lower}.com"

        # Fallback to web search
        return f"https://www.google.com/search?q={urllib.parse.quote_plus(clean)}"

    SEARCH_PLATFORMS: Dict[str, str] = {
        "google": "https://www.google.com/search?q={query}",
        "youtube": "https://www.youtube.com/results?search_query={query}",
        "yt": "https://www.youtube.com/results?search_query={query}",
        "amazon": "https://www.amazon.in/s?k={query}",
        "amazon.in": "https://www.amazon.in/s?k={query}",
        "amazon.com": "https://www.amazon.com/s?k={query}",
        "flipkart": "https://www.flipkart.com/search?q={query}",
        "github": "https://github.com/search?q={query}",
        "wikipedia": "https://en.wikipedia.org/wiki/Special:Search?search={query}",
        "wiki": "https://en.wikipedia.org/wiki/Special:Search?search={query}",
        "twitter": "https://x.com/search?q={query}",
        "x": "https://x.com/search?q={query}",
        "reddit": "https://www.reddit.com/search/?q={query}",
        "spotify": "https://open.spotify.com/search/{query}",
        "bing": "https://www.bing.com/search?q={query}",
        "duckduckgo": "https://duckduckgo.com/?q={query}",
        "ddg": "https://duckduckgo.com/?q={query}",
        "linkedin": "https://www.linkedin.com/search/results/all/?keywords={query}",
        "stackoverflow": "https://stackoverflow.com/search?q={query}",
        "yahoo": "https://search.yahoo.com/search?p={query}",
        "cricbuzz": "https://www.cricbuzz.com/search?q={query}",
        "medium": "https://medium.com/search?q={query}",
        "pinterest": "https://www.pinterest.com/search/pins/?q={query}",
        "quora": "https://www.quora.com/search?q={query}",
    }

    @classmethod
    def build_search_url(cls, platform: str, query: str) -> str:
        """Constructs a direct search URL for the specified platform and query."""
        if not query:
            return cls.resolve_url(platform) or "https://www.google.com"
        clean_plat = platform.lower().strip() if platform else "google"
        encoded = urllib.parse.quote_plus(query.strip())

        # Normalize platform aliases
        if clean_plat in ("chrome", "browser", "the browser", "web", "internet", "google chrome", "edge", "microsoft edge"):
            clean_plat = "google"
        elif clean_plat in ("yt", "you tube"):
            clean_plat = "youtube"
        elif clean_plat in ("wiki",):
            clean_plat = "wikipedia"

        if clean_plat in cls.SEARCH_PLATFORMS:
            return cls.SEARCH_PLATFORMS[clean_plat].format(query=encoded)

        # Check if platform is a known website in DEFAULT_URL_MAP
        base_url = cls.resolve_url(clean_plat)
        if base_url:
            domain = re.sub(r"^https?://(?:www\.)?", "", base_url).split("/")[0]
            return f"https://www.google.com/search?q=site%3A{domain}+{encoded}"

        # Default fallback to Google search
        return f"https://www.google.com/search?q={encoded}"


