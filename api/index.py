"""
Vercel serverless function entry point for FastAPI app
"""
from app.main import app

# Vercel expects a variable named 'app'
# FastAPI is ASGI compatible, which works with Vercel
handler = app