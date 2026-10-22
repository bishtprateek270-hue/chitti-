"""
Script to seed all important details, background, education, contact, skills,
and all GitHub & portfolio projects into Chitti's long-term persistent memory database.
"""

from src.memory.manager import MemoryManager
from src.memory.database import MemoryRecord

def seed_memories():
    m = MemoryManager()
    db = m.db

    # 1. Clean up garbage/test memories from previous voice experiments
    garbage_ids = [14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28]
    for gid in garbage_ids:
        try:
            db.delete_memory(gid)
            print(f"Cleaned up garbage memory ID: {gid}")
        except Exception:
            pass

    # 2. Structured Profile Details
    core_details = [
        # Identity & Creator
        {
            "content": "User's name is Prateek Singh Bisht.",
            "type": "identity",
            "importance": 5,
            "metadata": {"key": "user_name", "value": "Prateek Singh Bisht"}
        },
        {
            "content": "Prateek Singh Bisht is my creator and developer who built and designed Chitti.",
            "type": "identity",
            "importance": 5,
            "metadata": {"key": "creator", "value": "Prateek Singh Bisht"}
        },
        {
            "content": "User is an AI & Machine Learning Engineer specializing in Deep Learning, Computer Vision, LLMs, and Production MLOps.",
            "type": "professional_identity",
            "importance": 5,
            "metadata": {"key": "occupation", "value": "AI & Machine Learning Engineer"}
        },
        # Education & College
        {
            "content": "User is a 3rd-year B.Tech student (5th Semester) in Artificial Intelligence & Machine Learning at Panipat Institute of Engineering & Technology (PIET), Delhi-NCR (Batch 2024–2028).",
            "type": "education",
            "importance": 5,
            "metadata": {"key": "college", "value": "Panipat Institute of Engineering & Technology (PIET)"}
        },
        {
            "content": "User's academic record at PIET: Semester 1: 7.95 SGPA, Semester 2: 8.21 SGPA, Semester 3: 7.30 SGPA, Semester 4: 7.60 SGPA.",
            "type": "education",
            "importance": 4,
            "metadata": {"key": "academic_record", "value": "Sem 1: 7.95, Sem 2: 8.21, Sem 3: 7.30, Sem 4: 7.60"}
        },
        # Contact & Links
        {
            "content": "User is located in Delhi, India.",
            "type": "personal",
            "importance": 4,
            "metadata": {"key": "location", "value": "Delhi, India"}
        },
        {
            "content": "User's email address is bishtprateek270@gmail.com and phone number is +91 85955 52911.",
            "type": "contact",
            "importance": 4,
            "metadata": {"key": "contact", "email": "bishtprateek270@gmail.com", "phone": "+91 85955 52911"}
        },
        {
            "content": "User's GitHub profile is https://github.com/bishtprateek270-hue showcasing 14 repositories across AI, Full-Stack, and MLOps.",
            "type": "links",
            "importance": 5,
            "metadata": {"key": "github", "value": "https://github.com/bishtprateek270-hue"}
        },
        {
            "content": "User's personal portfolio website is live at https://bishtprateek270-hue.github.io/Portfolio/ and LinkedIn is https://linkedin.com/in/prateek-singh-bisht.",
            "type": "links",
            "importance": 5,
            "metadata": {"key": "portfolio", "value": "https://bishtprateek270-hue.github.io/Portfolio/"}
        },
        # Technical Stack
        {
            "content": "User's AI & Machine Learning tech stack: Python, PyTorch, Scikit-learn, OpenCV, XGBoost, Computer Vision, NLP, LLMs, RAG, MLflow, and Gemini API.",
            "type": "skill",
            "importance": 5,
            "metadata": {"key": "ai_skills", "value": "Python, PyTorch, Scikit-learn, OpenCV, XGBoost, Computer Vision, NLP, LLMs, RAG, MLflow"}
        },
        {
            "content": "User's Backend & Data stack: FastAPI, Flask, PostgreSQL, PostGIS, MongoDB, SQLite, Firebase, and REST APIs.",
            "type": "skill",
            "importance": 4,
            "metadata": {"key": "backend_skills", "value": "FastAPI, Flask, PostgreSQL, PostGIS, MongoDB, SQLite, Firebase"}
        },
        {
            "content": "User's Web & Mobile frontend stack: React 18/19, Next.js 14/16, TypeScript, JavaScript, Tailwind CSS, Flutter, Dart, and Streamlit.",
            "type": "skill",
            "importance": 4,
            "metadata": {"key": "frontend_skills", "value": "React, Next.js, TypeScript, Tailwind CSS, Flutter, Streamlit"}
        },
        {
            "content": "User's 3D & Geospatial visualization stack: Three.js, WebGL, deck.gl v9, MapLibre GL, and PostGIS for ISO 19152 3D volumetric cadastre.",
            "type": "skill",
            "importance": 4,
            "metadata": {"key": "3d_skills", "value": "Three.js, WebGL, deck.gl, MapLibre GL, PostGIS"}
        },
        {
            "content": "User's DevOps and tooling stack: Docker, Docker Compose, Git, GitHub, MLflow, Postman, Vercel, and Render.",
            "type": "skill",
            "importance": 4,
            "metadata": {"key": "devops_skills", "value": "Docker, Docker Compose, Git, GitHub, MLflow, Vercel, Render"}
        },
        # Work Experience & Achievements
        {
            "content": "User completed a 45-day Data Science & Machine Learning Internship at The Brainhub (July–August 2025), building ML feature engineering pipelines, model evaluations, and client analytics dashboards.",
            "type": "experience",
            "importance": 5,
            "metadata": {"key": "internship", "company": "The Brainhub", "role": "Data Science & ML Intern", "period": "July 2025 - August 2025"}
        },
        {
            "content": "User's hackathon achievements: Smart India Hackathon (SIH 2025) Internal Finalist at PIET, COER Manthan '25 Contender at COER University Roorkee, Hack KRMU 5.0 Contender at K.R. Mangalam University, and LogoLocha Design Award at SKLZ TECT.",
            "type": "achievement",
            "importance": 5,
            "metadata": {"key": "hackathons", "value": "SIH 2025 Finalist, COER Manthan '25, Hack KRMU 5.0, LogoLocha Design Award"}
        },
        {
            "content": "User's verified certifications: Google Cloud GenAI Studio (Aug 2026), Intro to Generative AI by Google Cloud (Aug 2026), IIT Bombay C Programming with 85% score (Dec 2024), Brainhub DS & ML Program (Aug 2025), and ICAT Certified Candidate (Aug 2026).",
            "type": "achievement",
            "importance": 4,
            "metadata": {"key": "certifications", "value": "Google Cloud GenAI Studio, IIT Bombay C Programming (85%), Brainhub DS & ML"}
        },
        # Projects
        {
            "content": "Project 1: Chitti — Personal AI Desktop Companion Robot. Built by Prateek Singh Bisht with Voice AI Brain, Persistent Long-Term Memory, Computer Vision Face Recognition, Multilingual Support (English, Hindi, Hinglish), and Controlled Laptop OS Automation. GitHub: https://github.com/bishtprateek270-hue/chitti-",
            "type": "project",
            "importance": 5,
            "metadata": {"key": "project_chitti", "name": "Chitti", "repo": "https://github.com/bishtprateek270-hue/chitti-"}
        },
        {
            "content": "Project 2: 3D-ULPIN — AI 3D Cadastral & Volumetric Land Registry System (ISO 19152 LADM). Modernizes 2D land parcels into multi-layer 3D volumetric digital twins with autonomous satellite footprint vectorization, Gemini Vision AI structural height classification, and 60 FPS WebGL strata slicing using React 18, TypeScript, deck.gl v9, MapLibre GL, Three.js, FastAPI, and PostGIS. Live: https://3-d-ulpin-mvp.vercel.app/ | GitHub: https://github.com/harsh3011dev-oops/3D-ULPIN-MVP",
            "type": "project",
            "importance": 5,
            "metadata": {"key": "project_3d_ulpin", "name": "3D-ULPIN", "live": "https://3-d-ulpin-mvp.vercel.app/", "repo": "https://github.com/harsh3011dev-oops/3D-ULPIN-MVP"}
        },
        {
            "content": "Project 3: KarigarAI (कारीगर AI) — Full-stack AI marketplace and WhatsApp commerce platform empowering Indian artisans to auto-generate multilingual catalogs from craft photos using Vision AI, fair labor-based pricing, printable exhibition QR price tags, and direct WhatsApp buyer ordering. Stack: React 19, Vite, FastAPI, Python, Vision AI. Live: https://karigar-ai-amber.vercel.app | GitHub: https://github.com/bishtprateek270-hue/karigarAI",
            "type": "project",
            "importance": 5,
            "metadata": {"key": "project_karigar_ai", "name": "KarigarAI", "live": "https://karigar-ai-amber.vercel.app", "repo": "https://github.com/bishtprateek270-hue/karigarAI"}
        },
        {
            "content": "Project 4: DocForensics AI — Enterprise multi-evidence document forensic analysis system detecting pixel-level tampering and micro-text alterations in PDFs and invoices using dual-stream deep learning (RGB + SRM residual stream and DocTamper with Focal Tversky Loss) and court-ready PDF audit reports. Stack: Next.js 14, FastAPI, PyTorch 2.0, TypeScript, OpenCV. Live: https://doc-forensics-ai.vercel.app | GitHub: https://github.com/bishtprateek270-hue/DocForensics-AI",
            "type": "project",
            "importance": 5,
            "metadata": {"key": "project_doc_forensics", "name": "DocForensics AI", "live": "https://doc-forensics-ai.vercel.app", "repo": "https://github.com/bishtprateek270-hue/DocForensics-AI"}
        },
        {
            "content": "Project 5: DoseBand — Industrial occupational safety IoT intelligence system for Hydrogen Sulfide (H₂S) toxic gas monitoring. Digitizes optical lead-acetate colorimetric dosimeter strips using computer vision image processing, detects ppm-hour dosages against DGMS/OSHA thresholds, and logs real-time geo-tagged worker telemetry. Stack: Flutter 3.x, Streamlit, OpenCV, Scikit-Learn, Python, SQLite. GitHub: https://github.com/bishtprateek270-hue/DoseBand",
            "type": "project",
            "importance": 5,
            "metadata": {"key": "project_doseband", "name": "DoseBand", "repo": "https://github.com/bishtprateek270-hue/DoseBand"}
        },
        {
            "content": "Project 6: AI Resume Analyzer & ATS Optimizer — Intelligent career acceleration tool parsing PDF resumes against target job descriptions, computing deterministic ATS match scores, providing LLM qualitative feedback, and recommending projects. Stack: Streamlit, Python, PostgreSQL, Google Gemini 2.5 Flash. Live: https://resume-analyzer-nejevwpy2bcvrvuag9schr.streamlit.app/ | GitHub: https://github.com/bishtprateek270-hue/resume-analyzer",
            "type": "project",
            "importance": 5,
            "metadata": {"key": "project_resume_analyzer", "name": "AI Resume Analyzer", "live": "https://resume-analyzer-nejevwpy2bcvrvuag9schr.streamlit.app/", "repo": "https://github.com/bishtprateek270-hue/resume-analyzer"}
        },
        {
            "content": "Project 7: RAGMind / AI-RAG-Platform — Enterprise Retrieval-Augmented Generation (RAG) platform supporting multi-format document indexing (.pdf, .docx, .pptx, .md) with vector search, source-cited Q&A chat with exact page numbers, executive summaries, AI flashcards, and interactive quizzes. Stack: Next.js 16, TypeScript, FastAPI, SQLite, Gemini API. Live: https://ai-rag-platform-sigma.vercel.app | GitHub: https://github.com/bishtprateek270-hue/AI-RAG-Platform",
            "type": "project",
            "importance": 5,
            "metadata": {"key": "project_ragmind", "name": "RAGMind / AI-RAG-Platform", "live": "https://ai-rag-platform-sigma.vercel.app", "repo": "https://github.com/bishtprateek270-hue/AI-RAG-Platform"}
        },
        {
            "content": "Project 8: ShikshaSetu — AI-powered EdTech study platform built with Next.js and Firebase, featuring automated study notes generation, dynamic PDF-to-interactive-quiz conversion, an adaptive doubt solver, and customized mock test pipelines. Stack: Next.js, TypeScript, Firebase, Gemini API. Live: https://shikshasetu-ten.vercel.app | GitHub: https://github.com/bishtprateek270-hue/ShikshaSetu",
            "type": "project",
            "importance": 5,
            "metadata": {"key": "project_shikshasetu", "name": "ShikshaSetu", "live": "https://shikshasetu-ten.vercel.app", "repo": "https://github.com/bishtprateek270-hue/ShikshaSetu"}
        },
        {
            "content": "Project 9: ChurnOps — Production-ready end-to-end MLOps pipeline and REST API for predicting customer churn risk with automated schema detection, hyperparameter tuning, model versioning with MLflow, containerized Docker deployment, and OpenAPI/Swagger test interface. Stack: FastAPI, XGBoost, Scikit-Learn, Docker, MLflow, Python. Live: https://churnops-u0bn.onrender.com | GitHub: https://github.com/bishtprateek270-hue/churnops",
            "type": "project",
            "importance": 5,
            "metadata": {"key": "project_churnops", "name": "ChurnOps", "live": "https://churnops-u0bn.onrender.com", "repo": "https://github.com/bishtprateek270-hue/churnops"}
        },
        {
            "content": "Project 10: SimplePay — Containerized 3-tier payment application dashboard featuring a React Single Page Application frontend, a Python Flask REST API backend, and persistent MongoDB storage orchestrated with Docker Compose. Stack: React, Flask, MongoDB, Docker, Docker Compose, REST API. Live: https://simplepay-946w.onrender.com | GitHub: https://github.com/bishtprateek270-hue/SimplePay",
            "type": "project",
            "importance": 5,
            "metadata": {"key": "project_simplepay", "name": "SimplePay", "live": "https://simplepay-946w.onrender.com", "repo": "https://github.com/bishtprateek270-hue/SimplePay"}
        },
        {
            "content": "Additional Projects: User has also built and published 'puzzle-app' (interactive puzzle web app deployed at https://puzzle-app-53uh.onrender.com), 'dsa' (C++ data structures and algorithms library), and 'ml-ops' (reproducible machine learning pipelines).",
            "type": "project",
            "importance": 4,
            "metadata": {"key": "additional_projects", "repos": "puzzle-app, dsa, ml-ops"}
        }
    ]

    print(f"\nSeeding {len(core_details)} core memories...")
    for item in core_details:
        success, msg, rec = m.remember(
            content=item["content"],
            memory_type=item["type"],
            importance=item["importance"],
            metadata=item["metadata"]
        )
        print(f"[{'OK' if success else 'FAIL'}] {item['metadata'].get('key')}: {msg}")

    print("\nVerifying stored memories count...")
    all_mems = db.list_all_memories()
    print(f"Total memories in database: {len(all_mems)}")
    print("\n--- Current High Importance Memories ---")
    for rec in all_mems:
        if rec.importance >= 4:
            print(f"[{rec.id}] ({rec.memory_type}) {rec.content[:90]}...")

if __name__ == "__main__":
    seed_memories()
