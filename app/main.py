import os
import shutil
import tempfile
import uuid
from typing import Optional, List, Dict
import secrets
import stat

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask
from pdf2docx import Converter
import fitz  # PyMuPDF for PDF rasterization
from PIL import Image
import pillow_heif
import io
import zipfile
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Register HEIF opener with Pillow for iPhone images
pillow_heif.register_heif_opener()

app = FastAPI(title="PDF to Word Converter")

# Mount static files (if any)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Jinja templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# In-memory sessions for reorder tool
REORDER_SESSIONS: Dict[str, str] = {}


def _secure_delete_file(filepath: str) -> None:
    """Securely delete a file by overwriting it with random data before removal."""
    if not filepath or not os.path.exists(filepath):
        return
    
    try:
        # Get file size
        filesize = os.path.getsize(filepath)
        
        # Overwrite file with random data
        with open(filepath, "ba+", buffering=0) as f:
            # Overwrite with random bytes
            f.seek(0)
            f.write(secrets.token_bytes(filesize))
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        
        # Remove the file
        os.remove(filepath)
    except Exception:
        # If secure deletion fails, try normal deletion
        try:
            os.remove(filepath)
        except Exception:
            pass


def _cleanup_files(*paths: str) -> None:
    """Cleanup files with secure deletion."""
    for path in paths:
        _secure_delete_file(path)


def _secure_cleanup_directory(directory: str) -> None:
    """Securely delete all files in a directory and remove it."""
    if not directory or not os.path.exists(directory):
        return
    
    try:
        # Securely delete all files in directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                _secure_delete_file(os.path.join(root, file))
        
        # Remove the directory
        shutil.rmtree(directory, ignore_errors=True)
    except Exception:
        # Fallback to regular removal
        try:
            shutil.rmtree(directory, ignore_errors=True)
        except Exception:
            pass


def _create_secure_temp_dir(prefix: str = "secure") -> str:
    """Create a secure temporary directory with restricted permissions."""
    # Add random UUID to prevent directory name prediction
    secure_prefix = f"{prefix}_{uuid.uuid4().hex[:8]}_{secrets.token_hex(4)}_"
    temp_dir = tempfile.mkdtemp(prefix=secure_prefix)
    
    # Set restrictive permissions (owner read/write/execute only)
    try:
        os.chmod(temp_dir, stat.S_IRWXU)  # 700 permissions
    except Exception:
        pass  # Permission setting might fail on some systems
    
    return temp_dir


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/pdf-to-word", response_class=HTMLResponse)
async def pdf_to_word_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/convert")
async def convert_pdf_to_docx(
    request: Request,
    pdf_file: UploadFile = File(..., description="Upload a PDF file"),
    start_page: Optional[int] = Form(None),
    end_page: Optional[int] = Form(None),
):
    # Basic validations
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    # Save upload to a temp file
    temp_dir = _create_secure_temp_dir(prefix="pdf2docx")
    input_pdf_path = os.path.join(temp_dir, pdf_file.filename)
    try:
        with open(input_pdf_path, "wb") as f:
            shutil.copyfileobj(pdf_file.file, f)
    except Exception as e:
        _cleanup_files(input_pdf_path)
        _secure_cleanup_directory(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Prepare output path
    base_name = os.path.splitext(os.path.basename(pdf_file.filename))[0]
    safe_base = base_name if base_name else f"converted-{uuid.uuid4().hex[:8]}"
    output_docx_path = os.path.join(temp_dir, f"{safe_base}.docx")

    # Standard conversion via pdf2docx
    try:
        # pdf2docx uses 0-based start, end is exclusive
        start = (start_page - 1) if start_page and start_page > 0 else 0
        end = end_page if end_page and end_page > 0 else None
        converter = Converter(input_pdf_path)
        try:
            converter.convert(output_docx_path, start=start, end=end)
        finally:
            converter.close()
    except Exception as e:
        _cleanup_files(input_pdf_path, output_docx_path)
        _secure_cleanup_directory(temp_dir)
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")

    # Return the file and cleanup afterwards
    filename_for_download = f"{safe_base}.docx"

    def _cleanup_all():
        _cleanup_files(input_pdf_path, output_docx_path)
        _secure_cleanup_directory(temp_dir)

    return FileResponse(
        output_docx_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename_for_download,
        background=BackgroundTask(_cleanup_all),
    )


@app.get("/pdf-to-image", response_class=HTMLResponse)
async def pdf_to_image_page(request: Request):
    return templates.TemplateResponse("pdf_to_image.html", {"request": request})


@app.post("/convert-image")
async def convert_pdf_to_images(
    request: Request,
    pdf_file: UploadFile = File(..., description="Upload a PDF file"),
    start_page: Optional[int] = Form(None),
    end_page: Optional[int] = Form(None),
    image_format: str = Form("png"),  # png | jpg | jpeg
    dpi: Optional[int] = Form(200),
):
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    # Normalize format
    fmt = image_format.lower()
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt not in {"png", "jpg"}:
        raise HTTPException(status_code=400, detail="image_format must be png or jpg/jpeg")

    dots_per_inch = dpi if dpi and dpi > 0 else 200

    temp_dir = _create_secure_temp_dir(prefix="pdf2img")
    input_pdf_path = os.path.join(temp_dir, pdf_file.filename)
    try:
        with open(input_pdf_path, "wb") as f:
            shutil.copyfileobj(pdf_file.file, f)
    except Exception as e:
        _cleanup_files(input_pdf_path)
        _secure_cleanup_directory(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Open PDF and render pages
    try:
        doc = fitz.open(input_pdf_path)
    except Exception as e:
        _cleanup_files(input_pdf_path)
        _secure_cleanup_directory(temp_dir)
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {e}")

    try:
        num_pages = doc.page_count
        start_idx = (start_page - 1) if start_page and start_page > 0 else 0
        end_idx_inclusive = (end_page - 1) if end_page and end_page > 0 else (num_pages - 1)
        if start_idx < 0:
            start_idx = 0
        if end_idx_inclusive >= num_pages:
            end_idx_inclusive = num_pages - 1
        if start_idx > end_idx_inclusive:
            raise HTTPException(status_code=400, detail="Invalid page range")

        base_name = os.path.splitext(os.path.basename(pdf_file.filename))[0]
        safe_base = base_name if base_name else f"converted-{uuid.uuid4().hex[:8]}"

        scale = float(dots_per_inch) / 72.0
        matrix = fitz.Matrix(scale, scale)

        generated_files = []
        for page_index in range(start_idx, end_idx_inclusive + 1):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            if fmt == "png":
                img_bytes = pix.tobytes("png")
            else:  # jpg
                mode = "RGB" if pix.n < 4 else "RGBA"
                image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                # Ensure no alpha for JPEG
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                buf = io.BytesIO()
                image.save(buf, format="JPEG", quality=90)
                img_bytes = buf.getvalue()

            ext = "png" if fmt == "png" else "jpg"
            out_path = os.path.join(temp_dir, f"{safe_base}-p{page_index + 1}.{ext}")
            with open(out_path, "wb") as out:
                out.write(img_bytes)
            generated_files.append(out_path)

    finally:
        doc.close()

    # Return single image or zip
    if len(generated_files) == 1:
        img_path = generated_files[0]
        filename_for_download = os.path.basename(img_path)

        def _cleanup_single():
            for p in [input_pdf_path] + generated_files:
                _cleanup_files(p)
            _secure_cleanup_directory(temp_dir)

        media_type = "image/png" if img_path.lower().endswith(".png") else "image/jpeg"
        return FileResponse(
            img_path,
            media_type=media_type,
            filename=filename_for_download,
            background=BackgroundTask(_cleanup_single),
        )

    # Zip multiple images
    zip_path = os.path.join(temp_dir, f"{safe_base}_images.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in generated_files:
            zf.write(p, arcname=os.path.basename(p))

    def _cleanup_zip():
        for p in [input_pdf_path] + generated_files + [zip_path]:
            _cleanup_files(p)
        _secure_cleanup_directory(temp_dir)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=os.path.basename(zip_path),
        background=BackgroundTask(_cleanup_zip),
    )


@app.get("/image-to-pdf", response_class=HTMLResponse)
async def image_to_pdf_page(request: Request):
    return templates.TemplateResponse("image_to_pdf.html", {"request": request})


@app.post("/convert-image-to-pdf")
async def convert_images_to_pdf(
    request: Request,
    images: List[UploadFile] = File(..., description="Upload one or more images"),
    filename: Optional[str] = Form(None),
):
    if not images:
        raise HTTPException(status_code=400, detail="Please upload at least one image")

    temp_dir = _create_secure_temp_dir(prefix="img2pdf")
    saved_paths: List[str] = []
    pil_images: List[Image.Image] = []
    try:
        # Save images to disk and load with PIL
        for idx, up in enumerate(images):
            name = up.filename or f"image_{idx+1}"
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                # attempt to infer extension from content-type
                ext = ".jpg" if (up.content_type and "jpeg" in up.content_type) else ".png"
                name = name + ext
            out_path = os.path.join(temp_dir, name)
            with open(out_path, "wb") as f:
                shutil.copyfileobj(up.file, f)
            saved_paths.append(out_path)

        # Load and normalize to RGB (PDF does not support alpha)
        for path in saved_paths:
            img = Image.open(path)
            if img.mode in ("RGBA", "LA"):
                img = img.convert("RGB")
            elif img.mode == "P":
                img = img.convert("RGB")
            pil_images.append(img)

        if not pil_images:
            raise HTTPException(status_code=400, detail="No valid images were provided")

        output_name = (filename or (os.path.splitext(os.path.basename(saved_paths[0]))[0] + "_merged")) + ".pdf"
        output_pdf_path = os.path.join(temp_dir, output_name)

        first, *rest = pil_images
        # Save as a PDF; resolution controls the embedded dpi for viewers
        first.save(output_pdf_path, format="PDF", save_all=True, append_images=rest, resolution=100.0)

    except Exception as e:
        for p in saved_paths:
            _cleanup_files(p)
        _secure_cleanup_directory(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to create PDF: {e}")

    def _cleanup_all():
        for p in saved_paths:
            _cleanup_files(p)
        _cleanup_files(output_pdf_path)
        _secure_cleanup_directory(temp_dir)

    return FileResponse(
        output_pdf_path,
        media_type="application/pdf",
        filename=os.path.basename(output_pdf_path),
        background=BackgroundTask(_cleanup_all),
    )


# ---- Reorder/Delete Pages ----
@app.get("/reorder-pdf", response_class=HTMLResponse)
async def reorder_pdf_page(request: Request):
    return templates.TemplateResponse("reorder_pdf.html", {"request": request})


@app.post("/reorder/init")
async def reorder_init(pdf_file: UploadFile = File(...)):
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    temp_dir = _create_secure_temp_dir(prefix="reorder")
    input_pdf_path = os.path.join(temp_dir, pdf_file.filename)
    thumbs_dir = os.path.join(temp_dir, "thumbs")
    os.makedirs(thumbs_dir, exist_ok=True)

    try:
        with open(input_pdf_path, "wb") as f:
            shutil.copyfileobj(pdf_file.file, f)
    except Exception as e:
        _secure_cleanup_directory(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    try:
        doc = fitz.open(input_pdf_path)
    except Exception as e:
        _secure_cleanup_directory(temp_dir)
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {e}")

    try:
        page_count = doc.page_count
        # Generate thumbnails at ~150px wide
        target_width = 180
        thumbs = []
        for i in range(page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap()
            scale = target_width / pix.width if pix.width else 1.0
            if scale <= 0 or scale > 4:
                scale = 1.0
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            img_path = os.path.join(thumbs_dir, f"p{i+1}.png")
            with open(img_path, "wb") as out:
                out.write(pix.tobytes("png"))
            thumbs.append({
                "page": i + 1,
                "thumb_url": f"/reorder/thumb/{{token}}/{i+1}",
            })
    finally:
        doc.close()

    token = uuid.uuid4().hex
    REORDER_SESSIONS[token] = temp_dir

    # Fill token into URLs
    for t in thumbs:
        t["thumb_url"] = t["thumb_url"].replace("{token}", token)

    return {"token": token, "page_count": page_count, "thumbs": thumbs}


@app.get("/reorder/thumb/{token}/{page}")
async def reorder_thumb(token: str, page: int):
    temp_dir = REORDER_SESSIONS.get(token)
    if not temp_dir:
        raise HTTPException(status_code=404, detail="Session not found")
    img_path = os.path.join(temp_dir, "thumbs", f"p{page}.png")
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Thumb not found")
    return FileResponse(img_path, media_type="image/png")


@app.post("/reorder/apply")
async def reorder_apply(request: Request):
    try:
        body = await request.json()
        token = body.get("token")
        order = body.get("order")  # list of 1-based page numbers in desired order
        if not token or not isinstance(order, list) or not all(isinstance(x, int) for x in order):
            raise ValueError("Invalid payload")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    temp_dir = REORDER_SESSIONS.get(token)
    if not temp_dir:
        raise HTTPException(status_code=404, detail="Session expired or not found")

    # Locate original PDF
    candidates = [p for p in os.listdir(temp_dir) if p.lower().endswith(".pdf")]
    if not candidates:
        raise HTTPException(status_code=404, detail="Original PDF missing")
    input_pdf_path = os.path.join(temp_dir, candidates[0])

    base = os.path.splitext(os.path.basename(input_pdf_path))[0]
    output_pdf_path = os.path.join(temp_dir, f"{base}_reordered.pdf")

    try:
        src = fitz.open(input_pdf_path)
        dst = fitz.open()
        for pnum in order:
            idx = pnum - 1
            if idx < 0 or idx >= src.page_count:
                continue
            dst.insert_pdf(src, from_page=idx, to_page=idx)
        dst.save(output_pdf_path)
        dst.close()
    except Exception as e:
        try:
            dst.close()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to reorder: {e}")
    finally:
        src.close()

    def _cleanup_all():
        try:
            _secure_cleanup_directory(temp_dir)
        finally:
            REORDER_SESSIONS.pop(token, None)

    return FileResponse(
        output_pdf_path,
        media_type="application/pdf",
        filename=os.path.basename(output_pdf_path),
        background=BackgroundTask(_cleanup_all),
    )


# ---- Compress PDF ----
@app.get("/compress-pdf", response_class=HTMLResponse)
async def compress_pdf_page(request: Request):
    return templates.TemplateResponse("compress_pdf.html", {"request": request})


@app.post("/compress")
async def compress_pdf(
    request: Request,
    pdf_file: UploadFile = File(..., description="Upload a PDF"),
    quality: int = Form(75),                # 40..95
    scale_percent: int = Form(100),         # 50..100
    lossless_cleanup: bool = Form(False),
):
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    # Clamp inputs
    quality = max(40, min(95, int(quality)))
    scale_percent = max(50, min(100, int(scale_percent)))

    temp_dir = _create_secure_temp_dir(prefix="compress")
    input_pdf_path = os.path.join(temp_dir, pdf_file.filename)
    try:
        with open(input_pdf_path, "wb") as f:
            shutil.copyfileobj(pdf_file.file, f)
    except Exception as e:
        _secure_cleanup_directory(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    base = os.path.splitext(os.path.basename(pdf_file.filename))[0]
    output_pdf_path = os.path.join(temp_dir, f"{base}_compressed.pdf")

    try:
        doc = fitz.open(input_pdf_path)
    except Exception as e:
        _secure_cleanup_directory(temp_dir)
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {e}")

    try:
        if lossless_cleanup:
            # Structural cleanup and stream compression only
            doc.save(
                output_pdf_path,
                deflate=True,
                garbage=4,
                clean=True,
                incremental=False,
            )
        else:
            # Recompress embedded images; skip masks/1-bit images
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                images = page.get_images(full=True)
                for img in images:
                    xref = img[0]
                    try:
                        extracted = doc.extract_image(xref)
                    except Exception:
                        continue
                    if not extracted:
                        continue
                    img_bytes = extracted.get("image")
                    ext = (extracted.get("ext", "").lower() or "")
                    width = extracted.get("width")
                    height = extracted.get("height")
                    # Heuristics: skip very small images or monochrome masks
                    if not img_bytes or not width or not height:
                        continue
                    if width * height < 32 * 32:
                        continue

                    try:
                        pil = Image.open(io.BytesIO(img_bytes))
                    except Exception:
                        continue

                    # Skip 1-bit or palettized masks
                    if pil.mode in ("1",):
                        continue

                    # Resample if needed
                    if scale_percent < 100:
                        new_w = max(1, int(pil.width * scale_percent / 100))
                        new_h = max(1, int(pil.height * scale_percent / 100))
                        pil = pil.resize((new_w, new_h), Image.LANCZOS)

                    # Convert to RGB and JPEG-compress
                    if pil.mode not in ("RGB", "L"):
                        pil = pil.convert("RGB")
                    buf = io.BytesIO()
                    pil.save(buf, format="JPEG", quality=quality, optimize=True)
                    new_bytes = buf.getvalue()

                    # Replace stream content
                    try:
                        doc.update_stream(xref, new_bytes)
                    except Exception:
                        # As fallback, ignore failures for this image
                        pass

            # Final save with cleanup
            doc.save(
                output_pdf_path,
                deflate=True,
                garbage=4,
                clean=True,
                incremental=False,
            )
    except Exception as e:
        # Ensure temp files are cleaned; close handled in finally
        _cleanup_files(input_pdf_path, output_pdf_path)
        _secure_cleanup_directory(temp_dir)
        raise HTTPException(status_code=500, detail=f"Compression failed: {e}")
    finally:
        try:
            doc.close()
        except Exception:
            pass

    def _cleanup_all():
        _cleanup_files(input_pdf_path, output_pdf_path)
        _secure_cleanup_directory(temp_dir)

    return FileResponse(
        output_pdf_path,
        media_type="application/pdf",
        filename=os.path.basename(output_pdf_path),
        background=BackgroundTask(_cleanup_all),
    )


@app.get("/flatten-pdf", response_class=HTMLResponse)
async def flatten_pdf_page(request: Request):
    return templates.TemplateResponse("flatten_pdf.html", {"request": request})


@app.post("/flatten")
async def flatten_pdf(
    request: Request,
    pdf_file: UploadFile = File(..., description="Upload a PDF file"),
    start_page: Optional[int] = Form(None),
    end_page: Optional[int] = Form(None),
    dpi: Optional[int] = Form(200),
):
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    temp_dir = _create_secure_temp_dir(prefix="flatten")
    input_pdf_path = os.path.join(temp_dir, pdf_file.filename)
    try:
        with open(input_pdf_path, "wb") as f:
            shutil.copyfileobj(pdf_file.file, f)
    except Exception as e:
        _cleanup_files(input_pdf_path)
        _secure_cleanup_directory(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    try:
        src = fitz.open(input_pdf_path)
    except Exception as e:
        _cleanup_files(input_pdf_path)
        _secure_cleanup_directory(temp_dir)
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {e}")

    base_name = os.path.splitext(os.path.basename(pdf_file.filename))[0]
    safe_base = base_name if base_name else f"flattened-{uuid.uuid4().hex[:8]}"
    output_pdf_path = os.path.join(temp_dir, f"{safe_base}_flattened.pdf")

    dots_per_inch = dpi if dpi and dpi > 0 else 200
    scale = float(dots_per_inch) / 72.0
    matrix = fitz.Matrix(scale, scale)

    try:
        dst = fitz.open()
        num_pages = src.page_count
        start_idx = (start_page - 1) if start_page and start_page > 0 else 0
        end_idx_inclusive = (end_page - 1) if end_page and end_page > 0 else (num_pages - 1)
        if start_idx < 0:
            start_idx = 0
        if end_idx_inclusive >= num_pages:
            end_idx_inclusive = num_pages - 1
        if start_idx > end_idx_inclusive:
            raise HTTPException(status_code=400, detail="Invalid page range")

        for page_index in range(start_idx, end_idx_inclusive + 1):
            spage = src.load_page(page_index)
            rect = spage.rect  # page rectangle in points
            pix = spage.get_pixmap(matrix=matrix, alpha=False)
            img_bytes = pix.tobytes("png")

            dpage = dst.new_page(width=rect.width, height=rect.height)
            dpage.insert_image(rect, stream=img_bytes)

        dst.save(output_pdf_path, deflate=True)
        dst.close()
    except Exception as e:
        try:
            dst.close()
        except Exception:
            pass
        _cleanup_files(input_pdf_path, output_pdf_path)
        _secure_cleanup_directory(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to flatten PDF: {e}")
    finally:
        src.close()

    def _cleanup_all():
        _cleanup_files(input_pdf_path, output_pdf_path)
        _secure_cleanup_directory(temp_dir)

    return FileResponse(
        output_pdf_path,
        media_type="application/pdf",
        filename=os.path.basename(output_pdf_path),
        background=BackgroundTask(_cleanup_all),
    )


# ---- Merge PDFs ----
@app.get("/merge-pdf", response_class=HTMLResponse)
async def merge_pdf_page(request: Request):
    return templates.TemplateResponse("merge_pdf.html", {"request": request})


@app.post("/merge")
async def merge_pdfs(
    request: Request,
    pdfs: list[UploadFile] = File(..., description="Upload PDFs in the desired order"),
    filename: Optional[str] = Form(None),
):
    if not pdfs or len(pdfs) < 2:
        raise HTTPException(status_code=400, detail="Please upload at least two PDF files to merge")

    temp_dir = _create_secure_temp_dir(prefix="merge")
    saved_paths: list[str] = []
    try:
        # Save inputs
        for idx, up in enumerate(pdfs):
            if not up.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"File #{idx+1} is not a PDF")
            out_path = os.path.join(temp_dir, f"{idx+1:03d}_" + up.filename)
            with open(out_path, "wb") as f:
                shutil.copyfileobj(up.file, f)
            saved_paths.append(out_path)

        # Merge
        dst = fitz.open()
        for path in saved_paths:
            src = fitz.open(path)
            dst.insert_pdf(src)
            src.close()

        out_name = (filename or "merged") + ".pdf"
        output_pdf_path = os.path.join(temp_dir, out_name)
        dst.save(output_pdf_path)
        dst.close()
    except Exception as e:
        for p in saved_paths:
            _cleanup_files(p)
        _secure_cleanup_directory(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to merge: {e}")

    def _cleanup_all():
        for p in saved_paths:
            _cleanup_files(p)
        _cleanup_files(output_pdf_path)
        _secure_cleanup_directory(temp_dir)

    return FileResponse(
        output_pdf_path,
        media_type="application/pdf",
        filename=os.path.basename(output_pdf_path),
        background=BackgroundTask(_cleanup_all),
    )


# ---- Split PDFs ----
@app.get("/split-pdf", response_class=HTMLResponse)
async def split_pdf_page(request: Request):
    return templates.TemplateResponse("split_pdf.html", {"request": request})


@app.post("/split")
async def split_pdf(
    request: Request,
    pdf_file: UploadFile = File(..., description="Upload a PDF"),
    split_all_pages: bool = Form(False),
    start_page: Optional[int] = Form(None),
    end_page: Optional[int] = Form(None),
):
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    temp_dir = _create_secure_temp_dir(prefix="split")
    input_pdf_path = os.path.join(temp_dir, pdf_file.filename)
    try:
        with open(input_pdf_path, "wb") as f:
            shutil.copyfileobj(pdf_file.file, f)
    except Exception as e:
        _cleanup_files(input_pdf_path)
        _secure_cleanup_directory(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    try:
        src = fitz.open(input_pdf_path)
    except Exception as e:
        _cleanup_files(input_pdf_path)
        _secure_cleanup_directory(temp_dir)
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {e}")

    base = os.path.splitext(os.path.basename(pdf_file.filename))[0]
    outputs: list[str] = []
    try:
        if split_all_pages:
            # Export each page to its own PDF
            for i in range(src.page_count):
                out_path = os.path.join(temp_dir, f"{base}-p{i+1}.pdf")
                dst = fitz.open()
                dst.insert_pdf(src, from_page=i, to_page=i)
                dst.save(out_path)
                dst.close()
                outputs.append(out_path)
        else:
            # Extract a range of pages into a single file
            if not start_page:
                start_page = 1
            if not end_page:
                end_page = src.page_count
            start_idx = max(0, start_page - 1)
            end_idx = min(src.page_count - 1, end_page - 1)
            if start_idx > end_idx:
                raise HTTPException(status_code=400, detail="Invalid page range")

            out_path = os.path.join(temp_dir, f"{base}-p{start_idx+1}-to-p{end_idx+1}.pdf")
            dst = fitz.open()
            dst.insert_pdf(src, from_page=start_idx, to_page=end_idx)
            dst.save(out_path)
            dst.close()
            outputs.append(out_path)
    except Exception as e:
        try:
            dst.close()
        except Exception:
            pass
        _cleanup_files(input_pdf_path)
        for p in outputs:
            _cleanup_files(p)
        _secure_cleanup_directory(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to split: {e}")
    finally:
        src.close()

    # Return either a single file or a zip
    if len(outputs) == 1:
        single = outputs[0]

        def _cleanup_single():
            _cleanup_files(input_pdf_path, single)
            _secure_cleanup_directory(temp_dir)

        return FileResponse(
            single,
            media_type="application/pdf",
            filename=os.path.basename(single),
            background=BackgroundTask(_cleanup_single),
        )

    zip_path = os.path.join(temp_dir, f"{base}_split.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in outputs:
            zf.write(p, arcname=os.path.basename(p))

    def _cleanup_zip():
        _cleanup_files(input_pdf_path, zip_path)
        for p in outputs:
            _cleanup_files(p)
        _secure_cleanup_directory(temp_dir)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=os.path.basename(zip_path),
        background=BackgroundTask(_cleanup_zip),
    )


# ---- Image Compression ----
@app.get("/compress-image", response_class=HTMLResponse)
async def compress_image_page(request: Request):
    return templates.TemplateResponse("compress_image.html", {"request": request})


@app.post("/compress-image")
async def compress_image(
    request: Request,
    image_files: List[UploadFile] = File(..., description="Upload image files"),
    quality: int = Form(85, ge=1, le=100),
    max_width: Optional[int] = Form(None),
    max_height: Optional[int] = Form(None),
    output_format: str = Form("original"),
):
    if not image_files:
        raise HTTPException(status_code=400, detail="Please upload at least one image file")
    
    # Debug: Print what we received
    print(f"Received {len(image_files)} files")
    for f in image_files:
        print(f"  - {f.filename} (content_type: {f.content_type}, size: {f.size if hasattr(f, 'size') else 'unknown'})")
    
    temp_dir = _create_secure_temp_dir(prefix="compress_img")
    compressed_files = []
    
    try:
        for image_file in image_files:
            # Skip empty files
            if not image_file.filename:
                print(f"Skipping empty filename")
                continue
                
            # Save uploaded file
            input_path = os.path.join(temp_dir, image_file.filename)
            print(f"Saving {image_file.filename} to {input_path}")
            with open(input_path, "wb") as f:
                shutil.copyfileobj(image_file.file, f)
            
            try:
                # Open the image and get original size
                img = Image.open(input_path)
                original_size = os.path.getsize(input_path)
                
                print(f"Original file size: {original_size / 1024:.1f} KB")
                
                # Strip EXIF data and metadata for smaller file size
                # Keep only the image data
                data = list(img.getdata())
                img_without_exif = Image.new(img.mode, img.size)
                img_without_exif.putdata(data)
                img = img_without_exif
                
                # Calculate target size based on quality percentage
                # Quality represents the target size as percentage of original
                target_size = int(original_size * (quality / 100))
                
                # Convert RGBA to RGB if needed for JPEG format
                if output_format == "jpeg" or (output_format == "original" and image_file.filename.lower().endswith(('.jpg', '.jpeg'))):
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # Create a white background
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                        img = background
                
                # Resize if needed
                if max_width or max_height:
                    current_width, current_height = img.size
                    
                    # Calculate new dimensions maintaining aspect ratio
                    if max_width and max_height:
                        ratio = min(max_width / current_width, max_height / current_height)
                    elif max_width:
                        ratio = max_width / current_width
                    else:
                        ratio = max_height / current_height
                    
                    if ratio < 1:  # Only resize if image is larger than max dimensions
                        new_width = int(current_width * ratio)
                        new_height = int(current_height * ratio)
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Determine output format
                if output_format == "original":
                    # Keep original format
                    ext = os.path.splitext(image_file.filename)[1].lower()
                    if ext in ['.heic', '.heif']:
                        # Convert HEIC/HEIF to JPEG
                        save_format = 'JPEG'
                        ext = '.jpg'
                    elif ext in ['.jpg', '.jpeg']:
                        save_format = 'JPEG'
                    elif ext == '.png':
                        save_format = 'PNG'
                    elif ext in ['.webp']:
                        save_format = 'WEBP'
                    else:
                        save_format = 'JPEG'
                        ext = '.jpg'
                else:
                    # Use specified format
                    save_format = output_format.upper()
                    ext = f'.{output_format.lower()}'
                
                # Generate output filename (sanitize for safety)
                base_name = os.path.splitext(image_file.filename)[0]
                # Replace spaces and special chars with underscores for safety
                safe_base_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in base_name)
                output_filename = f"{safe_base_name}_compressed{ext}"
                output_path = os.path.join(temp_dir, output_filename)
                
                print(f"Target size: {target_size / 1024:.1f} KB ({quality}% of original)")
                
                # Multi-pass progressive compression to achieve target size
                best_path = None
                best_size = float('inf')
                attempts = []
                
                # Helper function for compression with specific parameters
                def compress_image(img_to_compress, q, format_name, temp_path):
                    try:
                        if format_name == 'JPEG':
                            # Adjust subsampling based on quality
                            subsample = 2 if q < 50 else (1 if q < 80 else 0)
                            img_to_compress.save(temp_path, format='JPEG',
                                               quality=q,
                                               optimize=True,
                                               progressive=True,
                                               subsampling=subsample)
                        elif format_name == 'PNG':
                            # PNG doesn't have quality, so we adjust by reducing colors
                            if q < 30:
                                # Very aggressive: Convert to JPEG instead
                                if img_to_compress.mode == 'RGBA':
                                    bg = Image.new('RGB', img_to_compress.size, (255, 255, 255))
                                    bg.paste(img_to_compress, mask=img_to_compress.split()[-1])
                                    img_to_compress = bg
                                img_to_compress.save(temp_path, format='JPEG',
                                                   quality=q + 20,
                                                   optimize=True,
                                                   progressive=True)
                            elif q < 50:
                                # Reduce to limited colors
                                colors = int(32 + (q * 4))  # 32-200 colors
                                img_quantized = img_to_compress.convert('P', palette=Image.ADAPTIVE, colors=colors)
                                img_quantized.save(temp_path, format='PNG', optimize=True)
                            elif q < 70:
                                # Moderate color reduction
                                colors = int(200 + (q * 2))  # 200-340 colors
                                img_quantized = img_to_compress.convert('P', palette=Image.ADAPTIVE, colors=min(colors, 256))
                                img_quantized.save(temp_path, format='PNG', optimize=True)
                            else:
                                # High quality PNG
                                img_to_compress.save(temp_path, format='PNG', 
                                                   optimize=True, 
                                                   compress_level=9)
                        elif format_name == 'WEBP':
                            if q < 95:
                                img_to_compress.save(temp_path, format='WEBP',
                                                   quality=q,
                                                   method=6,
                                                   lossless=False)
                            else:
                                img_to_compress.save(temp_path, format='WEBP',
                                                   quality=100,
                                                   method=6,
                                                   lossless=True)
                        return os.path.getsize(temp_path)
                    except Exception as e:
                        print(f"Compression attempt failed: {e}")
                        return float('inf')
                
                # Binary search for optimal quality to reach target size
                min_q, max_q = 1, 100
                best_quality = quality
                
                # Initial attempt with user-specified quality
                temp_path = output_path + '.tmp'
                size = compress_image(img, quality, save_format, temp_path)
                if os.path.exists(temp_path):
                    if size <= target_size or quality >= 95:
                        # If we're already at or below target, or user wants high quality
                        shutil.move(temp_path, output_path)
                        best_path = output_path
                        best_size = size
                    else:
                        os.remove(temp_path)
                        
                        # Try to find optimal quality for target size
                        for attempt in range(8):  # Max 8 attempts
                            test_quality = (min_q + max_q) // 2
                            size = compress_image(img, test_quality, save_format, temp_path)
                            
                            if os.path.exists(temp_path):
                                attempts.append((test_quality, size))
                                
                                if size <= target_size:
                                    if size > best_size or best_size > target_size:
                                        # This is closer to target
                                        if best_path and os.path.exists(best_path):
                                            os.remove(best_path)
                                        shutil.move(temp_path, output_path)
                                        best_path = output_path
                                        best_size = size
                                        best_quality = test_quality
                                    else:
                                        os.remove(temp_path)
                                    
                                    if size > target_size * 0.85:  # Within 15% of target
                                        break
                                    min_q = test_quality + 1
                                else:
                                    os.remove(temp_path)
                                    max_q = test_quality - 1
                            
                            if min_q > max_q:
                                break
                
                # If no good result, use the best we found
                if not best_path or not os.path.exists(best_path):
                    # Fallback to original quality
                    compress_image(img, quality, save_format, output_path)
                    best_path = output_path
                    best_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                
                # Log compression results
                if best_size > 0:
                    compression_ratio = (1 - best_size / original_size) * 100
                    print(f"Compressed size: {best_size / 1024:.1f} KB")
                    print(f"Compression ratio: {compression_ratio:.1f}% reduction")
                    print(f"Final quality used: {best_quality}")
                    print(f"Attempts made: {attempts}")
                
                compressed_files.append(output_path)
                
            except Exception as e:
                # Log the error for debugging
                import traceback
                print(f"Error processing {image_file.filename}: {e}")
                print(traceback.format_exc())
                continue
        
        if not compressed_files:
            _secure_cleanup_directory(temp_dir)
            raise HTTPException(status_code=400, detail="No images could be processed")
        
        # Return single file or zip
        if len(compressed_files) == 1:
            single_file = compressed_files[0]
            
            def _cleanup_single():
                _secure_cleanup_directory(temp_dir)
            
            return FileResponse(
                single_file,
                media_type="application/octet-stream",
                filename=os.path.basename(single_file),
                background=BackgroundTask(_cleanup_single),
            )
        else:
            # Create zip file
            zip_path = os.path.join(temp_dir, "compressed_images.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in compressed_files:
                    zf.write(file_path, arcname=os.path.basename(file_path))
            
            def _cleanup_zip():
                _secure_cleanup_directory(temp_dir)
            
            return FileResponse(
                zip_path,
                media_type="application/zip",
                filename="compressed_images.zip",
                background=BackgroundTask(_cleanup_zip),
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        _secure_cleanup_directory(temp_dir)
        raise
    except Exception as e:
        _secure_cleanup_directory(temp_dir)
        raise HTTPException(status_code=500, detail=f"Compression failed: {e}")
