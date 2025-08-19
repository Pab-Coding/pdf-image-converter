# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAPI-based web application that provides PDF manipulation utilities including:
- PDF to Word (DOCX) conversion using pdf2docx
- PDF to image conversion using PyMuPDF (fitz)
- Image to PDF conversion using PIL
- PDF page reordering with drag-and-drop interface
- PDF compression with image quality settings
- PDF flattening (rasterization)
- PDF merging and splitting

## Architecture

- **Framework**: FastAPI with Jinja2 templates and static file serving
- **PDF Processing**: PyMuPDF (fitz) for core PDF operations, pdf2docx for Word conversion
- **Image Processing**: Pillow (PIL) for image manipulation
- **File Management**: All processing uses temporary directories with automatic cleanup via FastAPI BackgroundTasks
- **Session Management**: In-memory sessions for reorder functionality using REORDER_SESSIONS dict

## Key Components

- `app/main.py`: Main FastAPI application with all endpoints and business logic
- `app/templates/`: HTML templates with Tailwind CSS styling
- `app/static/`: Static assets directory (currently empty)

## Development Commands

### Quick Start
```bash
./run.sh
```
This creates a virtual environment, installs dependencies, and starts the development server with auto-reload.

### Manual Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Docker
```bash
docker build -t pdf2word .
docker run --rm -p 8000:8000 pdf2word
```

## Important Implementation Details

### Page Indexing Convention
- UI uses 1-based page numbers for user-friendliness
- Backend conversion functions use 0-based indexing
- Conversion happens at endpoint level: `start = (start_page - 1) if start_page and start_page > 0 else 0`

### File Cleanup Pattern
- All endpoints use temporary directories created with `tempfile.mkdtemp()`
- Cleanup is handled via `BackgroundTask` in FileResponse
- Helper function `_cleanup_files()` provides best-effort file removal

### Session Management for Reorder Tool
- Uses UUID tokens stored in `REORDER_SESSIONS` dict mapping to temp directory paths
- Generates thumbnail images for drag-and-drop interface
- Session cleanup happens when reordered PDF is downloaded

### Error Handling
- Consistent HTTPException usage with descriptive error messages
- Try-catch blocks around all file operations with proper cleanup
- Input validation for file types, page ranges, and parameters