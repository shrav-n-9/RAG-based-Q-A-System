import os
import pdfplumber

RAW_DIR = "data/raw"
PREPROCESSED_DIR = "data/processed"

def extract_text_from_pdf(pdf_path: str) -> str:
    #extracts texts from a pdf file using pdfplumber
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def preprocess_pdfs(raw_dir: str = RAW_DIR, out_dir: str = PREPROCESSED_DIR):
    #Converts all pdfs in raw_dir into plain text files inside out_dir
    os.makedirs(out_dir, exist_ok=True)
    for filename in os.listdir(raw_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(raw_dir, filename)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(out_dir, txt_filename)
            
            text = extract_text_from_pdf(pdf_path)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"Processed {filename} -> {txt_path}")
if __name__ == "__main__":
    preprocess_pdfs()