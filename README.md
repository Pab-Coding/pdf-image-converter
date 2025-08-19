# PDF & Image Processing Suite

A comprehensive FastAPI web application for PDF manipulation and image compression with a beautiful Tailwind UI.

## 🚀 Features

### PDF Tools
- **PDF to Word**: Convert PDF files to DOCX format
- **PDF to Image**: Export PDF pages as PNG or JPG images
- **Image to PDF**: Combine multiple images into a single PDF
- **Merge PDFs**: Combine multiple PDF files
- **Split PDF**: Extract pages or split into individual files
- **Reorder Pages**: Drag-and-drop interface to reorganize PDF pages
- **Compress PDF**: Reduce PDF file size with quality controls
- **Flatten PDF**: Rasterize PDF pages to make content non-selectable

### Image Tools
- **Smart Compression**: Progressive compression based on target file size
- **Format Conversion**: Convert between HEIC, JPG, PNG, WebP
- **Batch Processing**: Compress multiple images at once
- **iPhone Support**: Full HEIC/HEIF support for iPhone images
- **Resize Options**: Set maximum dimensions while maintaining aspect ratio
- **Quality Presets**: Tiny (20%), Small (40%), Balanced (75%), Quality (92%)

## 📋 Requirements
- Python 3.9+
- macOS, Linux, or Windows

## 🔧 Quick Start
```bash
./run.sh
```
Then open `http://127.0.0.1:8000`

If `run.sh` is not executable:
```bash
chmod +x run.sh && ./run.sh
```

## Manual Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## 🐳 Docker
```bash
docker build -t pdf-tools .
docker run --rm -p 8000:8000 pdf-tools
```

## 🚀 Deployment

### ⚠️ Important: Vercel Not Recommended
This application is **NOT suitable for Vercel** due to:
- File system limitations (read-only)
- 10-second timeout (too short for file processing)
- Memory constraints
- Binary dependency issues

### ✅ Recommended Platforms

#### Railway (Recommended)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
```

#### Render
1. Connect your GitHub repository
2. Choose "Web Service"
3. Use Docker for deployment
4. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

#### Google Cloud Run
```bash
# Build and deploy
gcloud run deploy pdf-tools \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300
```

#### Heroku
```bash
# Create app and deploy
heroku create your-app-name
git push heroku main
```

### GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin YOUR_GITHUB_URL
git push -u origin main
```

## 🛠️ Technology Stack
- **Backend**: FastAPI
- **PDF Processing**: PyMuPDF (fitz), pdf2docx
- **Image Processing**: Pillow, pillow-heif
- **Frontend**: HTML5, Tailwind CSS, Vanilla JavaScript
- **File Management**: Python tempfile with automatic cleanup

## 📝 Technical Notes
- All file processing uses temporary directories with automatic cleanup
- Page indexing: UI uses 1-based numbering, backend uses 0-based
- Image compression uses multi-pass algorithm to achieve target file sizes
- EXIF metadata is stripped from images for smaller file sizes
- Sessions for PDF reordering are stored in-memory

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
MIT License

## ⚡ Performance Tips
- Large PDFs may take time to process
- Batch operations work best with files under 10MB each
- For best compression results, use WebP format for photos
- PNG compression works best for graphics and screenshots