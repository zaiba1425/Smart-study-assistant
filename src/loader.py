import os
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation

def load_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    text = ""

    if ext == ".pdf":
        reader = PdfReader(filepath)
        for page in reader.pages:
            text += page.extract_text() + "\n"

    elif ext == ".docx":
        doc = Document(filepath)
        for para in doc.paragraphs:
            text += para.text + "\n"

    elif ext == ".pptx":
        prs = Presentation(filepath)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
    else:
        text = None

    return text
