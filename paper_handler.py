import fitz
import pdfplumber
from streamlit.runtime.uploaded_file_manager import UploadedFile

def extract_text(file: UploadedFile) -> str:
    file.seek(0)
    pdf_bytes = file.read()

    # Try pdfplumber first — better for research paper layouts
    text = _extract_with_pdfplumber(pdf_bytes)

    # Fall back to PyMuPDF if pdfplumber yields too little
    if len(text.strip()) < 200:
        text = _extract_with_pymupdf(pdf_bytes)

    return text

def _extract_with_pdfplumber(pdf_bytes: bytes) -> str:
    import io
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if page_text and page_text.strip():
                pages.append(page_text)
    return "\n".join(pages)

def _extract_with_pymupdf(pdf_bytes: bytes) -> str:
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            page_text = page.get_text("text")
            if page_text.strip():
                pages.append(page_text)
    return "\n".join(pages)