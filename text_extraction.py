from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# Path to your PDF file
pdf_path = "./Original PDF/HSC26-Bangla1st-Paper.pdf"

# Convert PDF to list of images
pages = convert_from_path(pdf_path, dpi=300)

full_text = ""

for i, page in enumerate(pages):
    print(f"OCR on page {i+1}...")
    # Run OCR on each page
    text = pytesseract.image_to_string(page, lang="ben+eng")
    full_text += text + "\n\n"

# Save the output to a text file
with open("./output_bangla_pytesseract_300.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("OCR completed. Output saved in output_bangla_pytesseract_300.txt")
