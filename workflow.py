import base64
import json
import os
from typing import TypedDict

from google import genai
from google.genai import types as gtypes
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from pdf_utils import pdf_to_page_images

load_dotenv()

DOC_TYPES = [
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other",
]

MODEL = "gemini-3.1-flash-lite-preview"


class ClaimState(TypedDict):
    claim_id: str
    pdf_bytes: bytes
    page_images: list[dict]
    classifications: dict[int, str]
    id_data: dict
    discharge_data: dict
    bill_data: dict
    result: dict


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _pages_for_type(state: ClaimState, doc_type: str) -> list[dict]:
    return [
        p for p in state["page_images"]
        if state["classifications"].get(p["page_num"]) == doc_type
    ]


def _image_part(page: dict) -> gtypes.Part:
    return gtypes.Part.from_bytes(
        data=base64.b64decode(page["base64"]),
        mime_type="image/png",
    )


def _parse_json(raw: str) -> dict | list:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

def segregator_node(state: ClaimState) -> dict:
    client = _client()
    page_images = pdf_to_page_images(state["pdf_bytes"])

    parts = []
    for p in page_images:
        parts.append(f"Page {p['page_num']}:")
        parts.append(_image_part(p))

    parts.append(
        "You are a medical claim document classifier.\n"
        "For each page shown above (labeled 'Page N:'), classify it into exactly one of these document types:\n"
        f"{', '.join(DOC_TYPES)}\n\n"
        "Rules:\n"
        "- claim_forms: insurance claim forms, pre-auth forms\n"
        "- cheque_or_bank_details: cheque images, bank account details, NEFT/RTGS forms\n"
        "- identity_document: Aadhaar, PAN, passport, driving license, insurance card, policy document\n"
        "- itemized_bill: hospital bills with line items and costs\n"
        "- discharge_summary: doctor's discharge summary, case summary\n"
        "- prescription: doctor's prescriptions, medication orders\n"
        "- investigation_report: lab reports, pathology, radiology, scan reports\n"
        "- cash_receipt: payment receipts, cash memos\n"
        "- other: anything that doesn't fit above\n\n"
        'Respond ONLY with valid JSON: {"1": "doc_type", "2": "doc_type", ...} '
        "where keys are page numbers as strings."
    )

    response = client.models.generate_content(model=MODEL, contents=parts)
    raw_map: dict[str, str] = _parse_json(response.text)
    classifications = {int(k): v for k, v in raw_map.items()}

    return {"page_images": page_images, "classifications": classifications}


def id_agent_node(state: ClaimState) -> dict:
    pages = _pages_for_type(state, "identity_document")
    if not pages:
        return {"id_data": {"note": "No identity document pages found"}}

    client = _client()
    parts = []
    for p in pages:
        parts.append(f"Page {p['page_num']}:")
        parts.append(_image_part(p))

    parts.append(
        "Extract all identity and insurance information from the document(s) above.\n"
        "Return a JSON object with these fields (use null if not found):\n"
        "{\n"
        '  "patient_name": string,\n'
        '  "date_of_birth": string,\n'
        '  "gender": string,\n'
        '  "id_numbers": {"aadhaar": string, "pan": string, "passport": string, "other": string},\n'
        '  "policy_number": string,\n'
        '  "insurance_company": string,\n'
        '  "sum_insured": string,\n'
        '  "policy_holder_name": string,\n'
        '  "address": string,\n'
        '  "contact_number": string,\n'
        '  "additional_details": {}\n'
        "}\n"
        "Respond ONLY with valid JSON."
    )

    response = client.models.generate_content(model=MODEL, contents=parts)
    return {"id_data": _parse_json(response.text)}

def discharge_agent_node(state: ClaimState) -> dict:
    pages = _pages_for_type(state, "discharge_summary")
    if not pages:
        return {"discharge_data": {"note": "No discharge summary pages found"}}

    client = _client()
    parts = []
    for p in pages:
        parts.append(f"Page {p['page_num']}:")
        parts.append(_image_part(p))

    parts.append(
        "Extract all information from the discharge summary document(s) above.\n"
        "Return a JSON object with these fields (use null if not found):\n"
        "{\n"
        '  "patient_name": string,\n'
        '  "age": string,\n'
        '  "gender": string,\n'
        '  "admission_date": string,\n'
        '  "discharge_date": string,\n'
        '  "length_of_stay_days": number,\n'
        '  "ward_type": string,\n'
        '  "hospital_name": string,\n'
        '  "treating_physician": string,\n'
        '  "primary_diagnosis": string,\n'
        '  "secondary_diagnoses": [string],\n'
        '  "procedures_performed": [string],\n'
        '  "condition_at_discharge": string,\n'
        '  "follow_up_instructions": string,\n'
        '  "additional_details": {}\n'
        "}\n"
        "Respond ONLY with valid JSON."
    )

    response = client.models.generate_content(model=MODEL, contents=parts)
    return {"discharge_data": _parse_json(response.text)}


def bill_agent_node(state: ClaimState) -> dict:
    pages = _pages_for_type(state, "itemized_bill")
    if not pages:
        return {"bill_data": {"note": "No itemized bill pages found"}}

    client = _client()
    parts = []
    for p in pages:
        parts.append(f"Page {p['page_num']}:")
        parts.append(_image_part(p))

    parts.append(
        "Extract all billing information from the itemized bill document(s) above.\n"
        "Return a JSON object with these fields (use null if not found):\n"
        "{\n"
        '  "hospital_name": string,\n'
        '  "patient_name": string,\n'
        '  "bill_number": string,\n'
        '  "bill_date": string,\n'
        '  "items": [\n'
        '    {"description": string, "quantity": number, "unit_price": number, "amount": number, "category": string}\n'
        "  ],\n"
        '  "subtotal": number,\n'
        '  "discounts": number,\n'
        '  "taxes": number,\n'
        '  "total_amount": number,\n'
        '  "amount_paid": number,\n'
        '  "balance_due": number,\n'
        '  "currency": string\n'
        "}\n"
        "Calculate total_amount as the sum of all item amounts if not explicitly stated.\n"
        "Respond ONLY with valid JSON."
    )

    response = client.models.generate_content(model=MODEL, contents=parts)
    return {"bill_data": _parse_json(response.text)}

def aggregator_node(state: ClaimState) -> dict:
    page_summary = {f"page_{n}": t for n, t in state["classifications"].items()}

    result = {
        "claim_id": state["claim_id"],
        "total_pages": len(state["page_images"]),
        "document_classification": page_summary,
        "documents_found": sorted(set(state["classifications"].values())),
        "extracted_data": {
            "identity": state.get("id_data", {}),
            "discharge_summary": state.get("discharge_data", {}),
            "itemized_bill": state.get("bill_data", {}),
        },
    }

    return {"result": result}


def build_graph():
    graph = StateGraph(ClaimState)

    graph.add_node("segregator", segregator_node)
    graph.add_node("id_agent", id_agent_node)
    graph.add_node("discharge_agent", discharge_agent_node)
    graph.add_node("bill_agent", bill_agent_node)
    graph.add_node("aggregator", aggregator_node)

    graph.add_edge(START, "segregator")
    graph.add_edge("segregator", "id_agent")
    graph.add_edge("segregator", "discharge_agent")
    graph.add_edge("segregator", "bill_agent")
    graph.add_edge("id_agent", "aggregator")
    graph.add_edge("discharge_agent", "aggregator")
    graph.add_edge("bill_agent", "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
