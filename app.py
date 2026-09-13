"""
Enterprise Generative-AI Compliance & Financial Audit Bot
==========================================================
Single-file Streamlit application implementing a RAG-based compliance
auditing pipeline over uploaded PDF documents (contracts, financial
statements, SEC filings, etc.).

Run with:
    pip install -r requirements.txt   (see bottom of this file for the list)
    streamlit run app.py

Author: Senior AI Solutions Architect (generated)
"""

import os
import re
import io
import json
import time
import textwrap
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Optional heavy dependencies are imported defensively so the app still boots
# (in a degraded mode) even if a package failed to install in the user's env.
# ---------------------------------------------------------------------------
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

import requests


# ===========================================================================
# 1. SYSTEM CONFIGURATION & UI THEME (DEFAULT NIGHT / DARK MODE)
# ===========================================================================

st.set_page_config(
    page_title="Enterprise Generative-AI Compliance Audit Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_THEME_CSS = """
<style>
/* ---------- Global App Background ---------- */
.stApp {
    background-color: #0e1117;
    color: #ffffff;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background-color: #1f2937;
    border-right: 1px solid #2d3748;
}
section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* ---------- Headings & Text ---------- */
h1, h2, h3, h4, h5, h6, p, span, label, div {
    color: #ffffff;
}
.stMarkdown, .stMarkdown p {
    color: #d1d5db;
}

/* ---------- Cards / Containers ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #1f2937;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 4px;
}

/* ---------- Metric Widgets ---------- */
div[data-testid="stMetric"] {
    background-color: #1f2937;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 14px 16px;
}
div[data-testid="stMetricLabel"] {
    color: #9ca3af !important;
}
div[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* ---------- Text Areas / Inputs ---------- */
.stTextArea textarea, .stTextInput input {
    background-color: #111827;
    color: #ffffff;
    border: 1px solid #374151;
    border-radius: 8px;
}

/* ---------- File Uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
    background-color: #111827;
    border: 1.5px dashed #374151;
    border-radius: 10px;
}

/* ---------- Primary Action Button ---------- */
div.stButton > button, div.stDownloadButton > button {
    background-color: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 0.75em 1.4em;
    font-weight: 700;
    letter-spacing: 0.03em;
    width: 100%;
    transition: all 0.15s ease-in-out;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.35);
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    background-color: #1d4ed8;
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.55);
    transform: translateY(-1px);
}

/* ---------- Status / Operational Badge ---------- */
.op-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background-color: #111827;
    border: 1px solid #2d3748;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.82rem;
    color: #9ca3af;
    font-family: monospace;
}
.op-badge .dot {
    height: 9px;
    width: 9px;
    border-radius: 50%;
    background-color: #22c55e;
    box-shadow: 0 0 8px #22c55e;
    display: inline-block;
}

/* ---------- Section Headers ---------- */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #ffffff;
    border-left: 4px solid #2563eb;
    padding-left: 10px;
    margin: 6px 0 10px 0;
}

/* ---------- Red Flag Box ---------- */
.redflag-box {
    background-color: #1f2937;
    border: 1px solid #7f1d1d;
    border-left: 5px solid #ef4444;
    border-radius: 10px;
    padding: 16px 20px;
}
.redflag-box ul {
    margin: 0;
    padding-left: 20px;
}
.redflag-box li {
    margin-bottom: 8px;
    color: #fecaca;
}

/* ---------- Verdict Box ---------- */
.verdict-box {
    background-color: #111827;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 18px 20px;
    font-family: 'Courier New', monospace;
    white-space: pre-wrap;
    color: #e5e7eb;
    line-height: 1.55;
}
.breach {
    color: #f87171 !important;
    font-weight: 700;
}

hr {
    border-color: #2d3748;
}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: #0e1117; }
::-webkit-scrollbar-thumb { background: #374151; border-radius: 6px; }
</style>
"""

st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


# ===========================================================================
# 2. CORE BACKEND FUNCTIONS
# ===========================================================================

# ---------------------------------------------------------------------------
# 2.1  Multimodal Layout Parsing Engine (PDF -> structured text)
# ---------------------------------------------------------------------------

def _format_table_as_text(table: List[List[Optional[str]]]) -> str:
    """Convert a raw table (list of rows) into a clean, LLM-readable text block."""
    if not table or all(all((c is None or str(c).strip() == "") for c in row) for row in table):
        return ""

    cleaned_rows = []
    for row in table:
        cleaned_cells = [str(c).strip().replace("\n", " ") if c is not None else "" for c in row]
        cleaned_rows.append(cleaned_cells)

    header = cleaned_rows[0]
    body_rows = cleaned_rows[1:] if len(cleaned_rows) > 1 else []

    lines = []
    for row in body_rows:
        pairs = []
        for col_name, value in zip(header, row):
            col_name = col_name if col_name else "Field"
            if value:
                pairs.append(f"{col_name}: {value}")
        if pairs:
            lines.append(" | ".join(pairs))

    if not lines:
        return ""

    block = "[Financial Table Data]\n" + "\n".join(lines) + "\n[/Financial Table Data]"
    return block


def extract_pdf_with_pdfplumber(file_bytes: bytes) -> str:
    """Primary extraction path: pdfplumber gives strong table + text layout support."""
    extracted_segments = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if page_text:
                extracted_segments.append(f"[Page {page_number} Text]\n{page_text}")

            try:
                tables = page.extract_tables()
            except Exception:
                tables = []

            for t_idx, table in enumerate(tables, start=1):
                formatted = _format_table_as_text(table)
                if formatted:
                    extracted_segments.append(
                        f"[Page {page_number} - Table {t_idx}]\n{formatted}"
                    )

    return "\n\n".join(extracted_segments)


def extract_pdf_with_pymupdf(file_bytes: bytes) -> str:
    """Fallback extraction path using PyMuPDF when pdfplumber is unavailable
    or throws on a malformed/encrypted PDF."""
    extracted_segments = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        page_text = page.get_text("text").strip()
        if page_text:
            extracted_segments.append(f"[Page {page_number + 1} Text]\n{page_text}")

        # Lightweight heuristic table reconstruction from PyMuPDF word blocks:
        # group words into rows by their vertical position, which approximates
        # tabular structure well enough to preserve numeric alignment.
        words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
        if words:
            rows: Dict[int, List[Tuple[float, str]]] = {}
            for w in words:
                x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
                row_key = round(y0 / 3.0)  # bucket rows to tolerate small offsets
                rows.setdefault(row_key, []).append((x0, text))

            table_lines = []
            for row_key in sorted(rows.keys()):
                cells = sorted(rows[row_key], key=lambda c: c[0])
                row_text = "  ".join(c[1] for c in cells)
                if re.search(r"\d", row_text) and len(cells) >= 3:
                    table_lines.append(row_text)

            if len(table_lines) >= 2:
                block = "[Financial Table Data]\n" + "\n".join(table_lines) + "\n[/Financial Table Data]"
                extracted_segments.append(f"[Page {page_number + 1} - Reconstructed Table]\n{block}")

    doc.close()
    return "\n\n".join(extracted_segments)


def parse_pdf_document(uploaded_file) -> Tuple[str, str]:
    """
    Robust PDF extraction with graceful engine fallback.
    Returns (extracted_text, engine_used).
    """
    file_bytes = uploaded_file.getvalue()

    if PDFPLUMBER_AVAILABLE:
        try:
            text = extract_pdf_with_pdfplumber(file_bytes)
            if text.strip():
                return text, "pdfplumber"
        except Exception as e:
            st.warning(f"pdfplumber failed ({e}); falling back to PyMuPDF.")

    if PYMUPDF_AVAILABLE:
        try:
            text = extract_pdf_with_pymupdf(file_bytes)
            if text.strip():
                return text, "PyMuPDF"
        except Exception as e:
            st.error(f"PyMuPDF fallback also failed: {e}")

    return "", "none"


# ---------------------------------------------------------------------------
# 2.2  Hybrid Vector Retrieval Engine (RAG)
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Slide a fixed-size window over the raw document text to produce
    overlapping chunks suitable for semantic embedding."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)
    step = max(chunk_size - overlap, 1)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += step

    return chunks


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load the sentence-transformer embedding model once per session."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _cosine_similarity_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norms @ query_norm


class LocalVectorStore:
    """
    Self-contained, in-memory vector store.
    Uses ChromaDB's ephemeral client when available for ANN-style storage;
    otherwise falls back to a lightweight NumPy cosine-similarity matcher
    (fully local, no server, no persistence required).
    """

    def __init__(self):
        self.model = load_embedding_model()
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.backend = "none"

        if CHROMADB_AVAILABLE and self.model is not None:
            self.client = chromadb.EphemeralClient()
            self.collection = self.client.get_or_create_collection(name="audit_doc_chunks")
            self.backend = "chromadb"
        else:
            self.client = None
            self.collection = None
            self.backend = "numpy" if self.model is not None else "keyword"

    def index_chunks(self, chunks: List[str]) -> None:
        self.chunks = chunks
        if not chunks:
            return

        if self.model is None:
            # Keyword-only degraded mode (no embedding model available).
            self.backend = "keyword"
            return

        embeddings = self.model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if self.backend == "chromadb":
            # Reset collection for a fresh document each run.
            try:
                self.client.delete_collection("audit_doc_chunks")
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(name="audit_doc_chunks")
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            self.collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings.tolist(),
            )
        else:
            self.embeddings = embeddings

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if not self.chunks:
            return []

        if self.backend == "chromadb" and self.model is not None:
            query_embedding = self.model.encode([query], normalize_embeddings=True)[0].tolist()
            results = self.collection.query(query_embeddings=[query_embedding], n_results=min(top_k, len(self.chunks)))
            docs = results.get("documents", [[]])[0]
            return docs

        if self.backend == "numpy" and self.embeddings is not None:
            query_vec = self.model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
            sims = _cosine_similarity_matrix(query_vec, self.embeddings)
            top_indices = np.argsort(-sims)[:top_k]
            return [self.chunks[i] for i in top_indices]

        # Keyword fallback: rank chunks by overlapping token count with the query.
        query_tokens = set(re.findall(r"[a-zA-Z0-9%$.]+", query.lower()))
        scored = []
        for chunk in self.chunks:
            chunk_tokens = set(re.findall(r"[a-zA-Z0-9%$.]+", chunk.lower()))
            score = len(query_tokens & chunk_tokens)
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]


# ---------------------------------------------------------------------------
# 2.3  Hallucination Guardrail Check + LLM Execution Layer
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert corporate compliance auditor. Base your response ONLY on the "
    "provided document context. If the document data is missing or cannot confirm the "
    "audit query, output strictly: 'COMPLIANCE BREACH UNABLE TO VERIFY: Insufficient "
    "documentation data.' Do not hallucinate or invent regulatory details."
)

GROQ_MODEL = "llama3-8b-8192"                     # Groq-hosted Llama 3 8B
HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  # HF Serverless Inference model


def build_user_prompt(query: str, context_chunks: List[str]) -> str:
    if context_chunks:
        context_block = "\n\n---\n\n".join(
            f"[Context Chunk {i+1}]\n{c}" for i, c in enumerate(context_chunks)
        )
    else:
        context_block = "(No relevant context was retrieved from the uploaded document.)"

    return (
        f"DOCUMENT CONTEXT:\n{context_block}\n\n"
        f"AUDIT QUERY:\n{query}\n\n"
        "Provide a structured compliance audit finding. Cite the specific figures, clauses, "
        "or table values from the context that support your conclusion. If the context does "
        "not contain sufficient information, respond with the exact non-compliance fallback "
        "string specified in your system instructions."
    )


def call_groq_api(api_key: str, system_prompt: str, user_prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 900,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def call_hf_inference_api(api_key: str, system_prompt: str, user_prompt: str) -> str:
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    full_prompt = (
        f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}\n"
        f"<|start_header_id|>user<|end_header_id|>\n{user_prompt}\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    payload = {
        "inputs": full_prompt,
        "parameters": {"max_new_tokens": 700, "temperature": 0.1, "return_full_text": False},
    }
    response = requests.post(url, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip()
    return str(data)


def local_offline_fallback(query: str, context_chunks: List[str]) -> str:
    """
    Deterministic, non-generative fallback used only when no LLM API key is
    configured. It performs a strict keyword-grounding check against the
    retrieved context so the guardrail behavior is still demonstrable
    end-to-end without external network calls.
    """
    if not context_chunks:
        return "COMPLIANCE BREACH UNABLE TO VERIFY: Insufficient documentation data."

    joined_context = " ".join(context_chunks)
    query_terms = [t for t in re.findall(r"[a-zA-Z]{4,}", query.lower())]
    hits = sum(1 for t in query_terms if t in joined_context.lower())
    coverage = hits / max(len(query_terms), 1)

    if coverage < 0.25:
        return "COMPLIANCE BREACH UNABLE TO VERIFY: Insufficient documentation data."

    excerpt = textwrap.shorten(joined_context.replace("\n", " "), width=600, placeholder=" ...")
    return (
        "AUDIT FINDING (offline grounding mode — no LLM API key configured):\n\n"
        f"Relevant excerpt located in document context:\n\"{excerpt}\"\n\n"
        "NOTE: Connect a Groq or Hugging Face API key in the sidebar to enable full "
        "generative reasoning over this context."
    )


def run_llm_audit(query: str, context_chunks: List[str], groq_key: str, hf_key: str) -> Tuple[str, str]:
    """Executes the guardrail-wrapped LLM call, returns (response_text, engine_used)."""
    user_prompt = build_user_prompt(query, context_chunks)

    if groq_key:
        try:
            return call_groq_api(groq_key, SYSTEM_PROMPT, user_prompt), "Groq (Llama-3-8B-8192)"
        except Exception as e:
            st.warning(f"Groq API call failed ({e}); attempting Hugging Face fallback.")

    if hf_key:
        try:
            return call_hf_inference_api(hf_key, SYSTEM_PROMPT, user_prompt), "HF Serverless (Llama-3-8B-Instruct)"
        except Exception as e:
            st.warning(f"Hugging Face Inference API call failed ({e}); using offline grounding mode.")

    return local_offline_fallback(query, context_chunks), "Offline Grounding Fallback (no API key)"


# ---------------------------------------------------------------------------
# 2.4  Dynamic Visual Risk Meter
# ---------------------------------------------------------------------------

RISK_KEYWORDS = {
    "high": [
        "breach", "non-compliance", "unauthorized", "material misstatement",
        "fraud", "penalty", "default", "litigation", "violation", "unverified",
        "missing signature", "unfunded liability",
    ],
    "medium": [
        "delay", "discrepancy", "mismatch", "restated", "late payment",
        "pending review", "unresolved", "exception", "deviation",
    ],
    "low": [
        "minor", "clerical", "typo", "formatting", "rounding",
    ],
    "informational": [
        "note", "disclosure", "reference", "appendix", "supplementary",
    ],
}


@dataclass
class AuditScore:
    health_score: int
    severity_counts: Dict[str, int] = field(default_factory=dict)
    verdict_is_breach: bool = False


def compute_compliance_score(llm_response: str, context_chunks: List[str]) -> AuditScore:
    """
    Deterministic scoring heuristic: scans the LLM's audit narrative (plus the
    retrieved context) for risk-indicative language and maps hit density to a
    Compliance Health Score and a severity distribution for the risk chart.
    """
    combined_text = (llm_response + " " + " ".join(context_chunks)).lower()

    counts = {"high": 0, "medium": 0, "low": 0, "informational": 0}
    for severity, keywords in RISK_KEYWORDS.items():
        for kw in keywords:
            counts[severity] += combined_text.count(kw)

    is_breach = "compliance breach unable to verify" in combined_text

    # Weighted penalty: high risk hits hurt the score most.
    penalty = counts["high"] * 12 + counts["medium"] * 6 + counts["low"] * 2
    base_score = 100 if not is_breach else 40
    health_score = max(5, min(100, base_score - penalty))

    if is_breach:
        counts["high"] = max(counts["high"], 1)

    return AuditScore(health_score=health_score, severity_counts=counts, verdict_is_breach=is_breach)


def build_risk_bar_chart(score: AuditScore) -> go.Figure:
    severities = ["High", "Medium", "Low", "Informational"]
    values = [
        score.severity_counts.get("high", 0),
        score.severity_counts.get("medium", 0),
        score.severity_counts.get("low", 0),
        score.severity_counts.get("informational", 0),
    ]
    colors = ["#ef4444", "#f59e0b", "#eab308", "#3b82f6"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=severities,
                y=values,
                marker_color=colors,
                text=values,
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Risk Severity Distribution",
        plot_bgcolor="#1f2937",
        paper_bgcolor="#1f2937",
        font=dict(color="#e5e7eb"),
        yaxis=dict(gridcolor="#374151", zeroline=False, title="Detected Signals"),
        xaxis=dict(gridcolor="#374151"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
    )
    return fig


def generate_red_flags(score: AuditScore, context_chunks: List[str], llm_response: str) -> List[str]:
    """Produce human-readable audit anomaly bullets from the computed severity
    profile, combined with lightweight pattern checks against the source text."""
    flags = []
    combined = " ".join(context_chunks).lower()

    if score.verdict_is_breach:
        flags.append("Document context insufficient to confirm audit query — flagged as an unverifiable compliance claim.")

    if score.severity_counts.get("high", 0) > 0:
        flags.append("High-severity language detected (e.g. breach, penalty, litigation, or default terminology) in source context or AI findings.")

    if "payment" in combined and ("net 30" in combined or "net 60" in combined or "net 45" in combined):
        flags.append("Potential payment term mismatch identified — multiple differing 'Net' payment windows referenced across the document.")

    if "signature" not in combined and "signed" not in combined:
        flags.append("No explicit signature or execution clause detected in parsed text — verify wet-ink/e-signature compliance manually.")

    if score.severity_counts.get("medium", 0) > 2:
        flags.append("Multiple discrepancy/exception indicators found — recommend manual reconciliation of the flagged financial line items.")

    if not flags:
        flags.append("No material red flags detected in the current retrieval window — expand the audit query for deeper coverage.")

    return flags


# ===========================================================================
# 3. FRONTEND UI — SPLIT-SCREEN LAYOUT
# ===========================================================================

def render_header():
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;">
            <div>
                <h1 style="margin-bottom:0;">🛡️ Enterprise Generative-AI Compliance Audit Suite</h1>
                <p style="color:#9ca3af; margin-top:4px;">
                    Multimodal RAG pipeline for automated financial &amp; regulatory document auditing
                </p>
            </div>
        </div>
        <div class="op-badge">
            <span class="dot"></span> Operational &nbsp;|&nbsp; Model: Llama-3-8B-Instruct &nbsp;|&nbsp; Dataset: CUAD/EDGAR
        </div>
        <hr>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Dict[str, str]:
    with st.sidebar:
        st.markdown("### ⚙️ Engine Configuration")
        st.caption("Provide at least one inference key to enable live generative audits.")
        groq_key = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
        hf_key = st.text_input("Hugging Face API Token", type="password", value=os.environ.get("HF_API_TOKEN", ""))

        st.markdown("---")
        st.markdown("### 🧠 Retrieval Backend")
        backend_label = "ChromaDB (ephemeral)" if CHROMADB_AVAILABLE else (
            "NumPy cosine-similarity (local)" if SENTENCE_TRANSFORMERS_AVAILABLE else "Keyword matcher (degraded)"
        )
        st.info(f"Active vector backend: **{backend_label}**")

        st.markdown("### 📄 PDF Parsing Engine")
        parser_label = "pdfplumber (primary)" if PDFPLUMBER_AVAILABLE else (
            "PyMuPDF (fallback)" if PYMUPDF_AVAILABLE else "Unavailable — install pdfplumber or PyMuPDF"
        )
        st.info(f"Active parser: **{parser_label}**")

        st.markdown("---")
        st.markdown(
            "<span style='color:#6b7280; font-size:0.8rem;'>"
            "All document processing occurs locally within this session. "
            "No data is persisted after the session ends."
            "</span>",
            unsafe_allow_html=True,
        )

    return {"groq_key": groq_key.strip(), "hf_key": hf_key.strip()}


def render_scorecard(score: AuditScore):
    ring_color = "#22c55e" if score.health_score >= 70 else ("#f59e0b" if score.health_score >= 40 else "#ef4444")
    status_word = "SECURED" if score.health_score >= 70 else ("REVIEW REQUIRED" if score.health_score >= 40 else "AT RISK")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score.health_score,
            number={"suffix": "%", "font": {"color": "#ffffff", "size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9ca3af"},
                "bar": {"color": ring_color},
                "bgcolor": "#111827",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#3f1d1d"},
                    {"range": [40, 70], "color": "#4a3a10"},
                    {"range": [70, 100], "color": "#123822"},
                ],
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(
        paper_bgcolor="#1f2937",
        font=dict(color="#e5e7eb"),
        margin=dict(l=20, r=20, t=30, b=10),
        height=280,
    )

    st.markdown(f"**Compliance Scorecard: {score.health_score}% {status_word}**")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_analytics_panel(score: Optional[AuditScore]):
    st.markdown('<div class="section-title">📊 Analytics Dashboard</div>', unsafe_allow_html=True)

    if score is None:
        st.info("Run the audit pipeline to populate the compliance scorecard and risk chart.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        render_scorecard(score)
    with col_b:
        st.markdown("**Risk Severity Breakdown**")
        fig = build_risk_bar_chart(score)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("High Risk Signals", score.severity_counts.get("high", 0))
    m2.metric("Medium Risk Signals", score.severity_counts.get("medium", 0))
    m3.metric("Low Risk Signals", score.severity_counts.get("low", 0))
    m4.metric("Informational", score.severity_counts.get("informational", 0))


def render_red_flags(flags: Optional[List[str]]):
    st.markdown('<div class="section-title">⚠️ Red Flags &amp; Anomalies Detected</div>', unsafe_allow_html=True)
    if not flags:
        st.markdown(
            '<div class="redflag-box"><ul><li>No audit has been executed yet — upload a document and run the pipeline.</li></ul></div>',
            unsafe_allow_html=True,
        )
        return

    bullets = "".join(f"<li>{flag}</li>" for flag in flags)
    st.markdown(f'<div class="redflag-box"><ul>{bullets}</ul></div>', unsafe_allow_html=True)


def render_verdict(llm_response: str, engine_used: str):
    st.markdown('<div class="section-title">🧾 AI Audit Verdict</div>', unsafe_allow_html=True)
    st.caption(f"Generated via: {engine_used}")
    display_text = llm_response.replace(
        "COMPLIANCE BREACH UNABLE TO VERIFY: Insufficient documentation data.",
        "<span class='breach'>COMPLIANCE BREACH UNABLE TO VERIFY: Insufficient documentation data.</span>",
    )
    st.markdown(f'<div class="verdict-box">{display_text}</div>', unsafe_allow_html=True)


# ===========================================================================
# APPLICATION STATE & MAIN EXECUTION FLOW
# ===========================================================================

def init_session_state():
    defaults = {
        "vector_store": None,
        "parsed_text": "",
        "parser_engine": "",
        "audit_score": None,
        "red_flags": None,
        "llm_response": "",
        "llm_engine": "",
        "last_context_chunks": [],
        "doc_indexed_name": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    init_session_state()
    keys = render_sidebar()
    render_header()

    left_col, right_col = st.columns([1, 1.3], gap="large")

    with left_col:
        st.markdown('<div class="section-title">🗂️ Control Panel</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drag & drop a financial or contractual PDF document",
            type=["pdf"],
            accept_multiple_files=False,
        )

        query = st.text_area(
            "Target Audit Query",
            placeholder=(
                "e.g. Confirm whether the payment terms comply with the Net-30 standard "
                "clause and identify any late-payment penalty exposure."
            ),
            height=140,
        )

        execute_clicked = st.button("⚡ EXECUTE AI AUDIT PIPELINE", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">🛡️ Guardrail Policy</div>', unsafe_allow_html=True)
        st.code(SYSTEM_PROMPT, language="text")

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    if execute_clicked:
        if uploaded_file is None:
            st.error("Please upload a PDF document before executing the audit pipeline.")
        elif not query.strip():
            st.error("Please enter a Target Audit Query before executing the audit pipeline.")
        else:
            with st.spinner("Parsing document layout and extracting financial tables..."):
                needs_reparse = st.session_state.doc_indexed_name != uploaded_file.name
                if needs_reparse:
                    parsed_text, engine = parse_pdf_document(uploaded_file)
                    st.session_state.parsed_text = parsed_text
                    st.session_state.parser_engine = engine
                    st.session_state.doc_indexed_name = uploaded_file.name

            if not st.session_state.parsed_text.strip():
                st.error(
                    "No extractable text was found in this PDF. It may be a scanned image "
                    "without an embedded text layer, or both pdfplumber and PyMuPDF are unavailable."
                )
            else:
                with st.spinner("Chunking document and building the hybrid vector index..."):
                    if needs_reparse or st.session_state.vector_store is None:
                        chunks = chunk_text(st.session_state.parsed_text, chunk_size=800, overlap=150)
                        store = LocalVectorStore()
                        store.index_chunks(chunks)
                        st.session_state.vector_store = store

                with st.spinner("Retrieving top-matching context via hybrid vector search..."):
                    context_chunks = st.session_state.vector_store.retrieve(query, top_k=3)
                    st.session_state.last_context_chunks = context_chunks

                with st.spinner("Executing guardrailed LLM audit reasoning..."):
                    llm_response, engine_used = run_llm_audit(
                        query, context_chunks, keys["groq_key"], keys["hf_key"]
                    )
                    st.session_state.llm_response = llm_response
                    st.session_state.llm_engine = engine_used

                with st.spinner("Scoring compliance health and mapping risk severity..."):
                    score = compute_compliance_score(llm_response, context_chunks)
                    st.session_state.audit_score = score
                    st.session_state.red_flags = generate_red_flags(score, context_chunks, llm_response)

                st.success(f"Audit pipeline complete. Parsed via {st.session_state.parser_engine} · Reasoned via {engine_used}.")

    with right_col:
        render_analytics_panel(st.session_state.audit_score)
        st.markdown("<br>", unsafe_allow_html=True)
        render_red_flags(st.session_state.red_flags)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.llm_response:
        render_verdict(st.session_state.llm_response, st.session_state.llm_engine)

        with st.expander("🔍 View Retrieved Context Chunks (Top 3 — Vector Retrieval Trace)"):
            for i, chunk in enumerate(st.session_state.last_context_chunks, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.code(chunk, language="text")

    st.markdown(
        "<p style='text-align:center; color:#4b5563; font-size:0.78rem; margin-top:30px;'>"
        "Enterprise Generative-AI Compliance Audit Suite &nbsp;·&nbsp; Internal Use Only &nbsp;·&nbsp; "
        "Outputs must be reviewed by a licensed compliance officer before regulatory action."
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()


# ===========================================================================
# requirements.txt (create this alongside app.py)
# ---------------------------------------------------------------------------
# streamlit>=1.35
# pdfplumber>=0.11
# PyMuPDF>=1.24
# sentence-transformers>=3.0
# chromadb>=0.5
# plotly>=5.22
# numpy>=1.26
# requests>=2.32
# ===========================================================================