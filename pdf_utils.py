import base64
import io
import pymupdf
from PIL import Image


def pdf_to_page_images(pdf_bytes: bytes, dpi: int = 150) -> list[dict]:
    """Convert each PDF page to a base64-encoded PNG image."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    mat = pymupdf.Matrix(dpi / 72, dpi / 72)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
        img_bytes = pix.tobytes("png")

        # Resize if image is too large (Claude limit ~5MB)
        img = Image.open(io.BytesIO(img_bytes))
        max_dim = 1568
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

        pages.append({
            "page_num": page_num + 1,
            "base64": base64.standard_b64encode(img_bytes).decode("utf-8"),
        })

    doc.close()
    return pages
