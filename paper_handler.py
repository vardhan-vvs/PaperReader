import pdfplumber

def extract_text(pdf_path):
    chunks = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                chunks += text
    return chunks