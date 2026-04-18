import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

load_dotenv()

from workflow import ClaimState, get_graph

app = FastAPI(
    title="Claim Processing API",
    description="AI-powered insurance claim document extraction using LangGraph + Gemini",
    version="1.0.0",
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/process")
async def process_claim(
    claim_id: str = Form(...),
    file: UploadFile = File(...),
) -> JSONResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    print(os.environ.get("GEMINI_API_KEY"))

    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not configured. Create a .env file with your key.",
        )

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    initial_state: ClaimState = {
        "claim_id": claim_id,
        "pdf_bytes": pdf_bytes,
        "page_images": [],
        "classifications": {},
        "id_data": {},
        "discharge_data": {},
        "bill_data": {},
        "result": {},
    }

    try:
        graph = get_graph()
        final_state = await graph.ainvoke(initial_state)
        return JSONResponse(content=final_state["result"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
