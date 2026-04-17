import asyncio
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("GEMINI_API_KEY"):
    sys.exit("Set GEMINI_API_KEY in .env first")

from workflow import ClaimState, get_graph


async def main():
    pdf_path = os.path.join(os.path.dirname(__file__), "final_image_protected.pdf")
    if not os.path.exists(pdf_path):
        sys.exit(f"PDF file not found at {pdf_path}")
        
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    initial: ClaimState = {
        "claim_id": "TEST-001",
        "pdf_bytes": pdf_bytes,
        "page_images": [],
        "classifications": {},
        "id_data": {},
        "discharge_data": {},
        "bill_data": {},
        "result": {},
    }

    print("Running claim processing pipeline...")
    graph = get_graph()
    final = await graph.ainvoke(initial)

    print(json.dumps(final["result"], indent=2))


asyncio.run(main())
