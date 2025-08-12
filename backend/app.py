from flask import Flask, request, jsonify, g
from flask_cors import CORS
from dotenv import load_dotenv
import logging
import re
import os
import uuid
import unicodedata
from collections import deque, defaultdict
from datetime import datetime
import pandas as pd

from langchain_community.embeddings import HuggingFaceEmbeddings   # la fel ca în ingest.py
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.callbacks import get_openai_callback


# ------------------------------- Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log', encoding='utf-8'),
              logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__)
CORS(app)

# Memorie conversațională per sesiune (în RAM)
conversation_history = {}
# Stare de subiect pe sesiune (topic + dacă e A.R. la plic intern)
session_state = {}  # {session_id: {"topic": str, "ar": bool}}

# Tracking costuri
usage_data = defaultdict(list)
EXCEL_FILE = "api_costs.xlsx"

# Prețuri — exemplu
INPUT_COST_PER_TOKEN  = 5e-6    # $5 / 1M
OUTPUT_COST_PER_TOKEN = 15e-6   # $15 / 1M

# ------------------------------- Config fișiere locale
DATA_DIR = "/mnt/data"
FILE_CORESC_INT = os.path.join(DATA_DIR, "Tarife - Corespondenţă internă.txt")
FILE_CORESC_INT_AR = os.path.join(DATA_DIR, "Corespondenţă internă cu confirmare de primire (A.R.).txt")
FILE_PACHET_MIC_INT_NEPRIO = os.path.join(DATA_DIR, "Pachet_Mic_Intern_Neprioritar.txt")

FILE_CORESC_INTL = os.path.join(DATA_DIR, "Corespondenţă internaţională.txt")
FILE_CORESC_INTL_AR = os.path.join(DATA_DIR, "Corespondenţă internaţională cu confirmare de primire (A.R.).txt")
FILE_PACHET_MIC_INTRACOM_AR = os.path.join(DATA_DIR, "Pachet mic internațional intracomunitar prioritar.txt")
FILE_PACHET_MIC_EXTRACOM_AR = os.path.join(DATA_DIR, "Pachet mic extracomunitar cu confirmare de primire (A.R.).txt")

# Triggere de tarifare (pentru a adăuga nota cu calculatorul)
PRICE_TRIGGERS = [
    "cat costa", "cât costă", "cost", "tarif", "preț", "pret",
    "kg", "kilograme", "colet", "expediere", "trimit", "trimitere",
    "ems", "sub 10 kg", "peste 10 kg",
    "plic", "scrisoare", "corespondență", "corespondenta", "carte poștală", "carte postala",
    "confirmare de primire", "aviz de primire", "a.r.", "a.r", "ar",
    "pachet mic", "small packet"
]
PRICE_NOTE = ("Pentru a calcula cu serviciile suplimentare, folosește calculatorul "
              "de tarife de pe site: https://www.posta-romana.ro/calculator-de-tarife.html")

def is_price_intent(q: str) -> bool:
    t = (q or "").lower()
    return any(k in t for k in PRICE_TRIGGERS)


# ------------------------------- Sesiune
@app.before_request
def init_session():
    data = request.get_json(silent=True) or {}
    g.json_body = data
    sid = data.get("session_id") or str(uuid.uuid4())
    g.session_id = sid
    if sid not in conversation_history:
        conversation_history[sid] = deque(maxlen=5)
    if sid not in session_state:
        session_state[sid] = {"topic": None, "ar": False}

def get_context():
    return "\n".join(conversation_history[g.session_id])

def save_to_excel():
    try:
        try:
            existing_df = pd.read_excel(EXCEL_FILE)
        except FileNotFoundError:
            existing_df = pd.DataFrame()

        current_df = pd.DataFrame(usage_data)
        combined_df = (pd.concat([existing_df, current_df], ignore_index=True)
                       if not existing_df.empty else current_df)
        combined_df.to_excel(EXCEL_FILE, index=False)
        logger.info(f"Datele au fost salvate în {EXCEL_FILE}")
        usage_data.clear()
    except Exception as e:
        logger.error(f"Eroare la salvarea în Excel: {str(e)}")

def calculate_cost(prompt_tokens, completion_tokens):
    input_cost = prompt_tokens * INPUT_COST_PER_TOKEN
    output_cost = completion_tokens * OUTPUT_COST_PER_TOKEN
    return input_cost, output_cost, input_cost + output_cost


# ------------------------------- Inițializare RAG
try:
    logger.info("Încărcare embeddings/vectorstore/LLM...")

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=2000
    )

    extraction_prompt = PromptTemplate(
        template="""
        [Context]
        {context}

        [Sarcină]
        Extrage toate informațiile relevante pentru: {question}
        Include detalii, condiții și excepții.

        [Reguli prioritare]
        - Dacă întrebarea este despre **colete externe** și **greutatea > 10 kg**, IGNORĂ tarifele EMS
          și folosește DOAR secțiunea 'Tarife colete externe peste 10 kg (TVA inclus)'.
        - Dacă utilizatorul specifică explicit 'EMS', folosește tarifele EMS.

        [Format]
        - Listă a punctelor relevante
        - Menționează dacă există informații incomplete

        [Rezultat]
        """,
        input_variables=["context", "question"]
    )

    synthesis_prompt = PromptTemplate(
        template="""
        [Informații găsite]
        {extracted_info}

        [Întrebare]
        {question}

        [Sarcină]
        1) Răspuns clar și concis bazat pe context
        2) Ton prietenos, conversațional
        3) Structurat în puncte când e util
        4) Folosește contextul conversației, dacă e cazul
        5) Nu inventa cifre dacă nu sunt în context

        [Răspuns]
        """,
        input_variables=["extracted_info", "question"]
    )

    extraction_chain = LLMChain(llm=llm, prompt=extraction_prompt)
    synthesis_chain = LLMChain(llm=llm, prompt=synthesis_prompt)
    qa_chain = StuffDocumentsChain(
        llm_chain=extraction_chain,
        document_variable_name="context",
        output_key="extracted_info"
    )

    logger.info("Sistem gata")
except Exception as e:
    logger.error(f"Eroare inițializare: {str(e)}", exc_info=True)
    raise


# ------------------------------- Helpers NL / determinist

def strip_diacritics(s: str) -> str:
    n = unicodedata.normalize('NFD', s or "")
    return ''.join(c for c in n if unicodedata.category(c) != 'Mn')

def fmt_lei(x: float) -> str:
    return f"{x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')

def read_file_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Nu pot citi fișierul: {path} -> {e}")
        return "Nu am putut citi tarifele din fișierul specificat."

def answer_from_file(path: str, header: str = None) -> str:
    body = read_file_text(path)
    return (f"{header}\n\n{body}".strip() if header else body)

def parse_weight_country(q: str):
    q_l = (q or "").lower().strip()
    m_w = re.search(r'(\d+[.,]?\d*)\s*kg', q_l)
    weight = float(m_w.group(1).replace(',', '.')) if m_w else None
    m_c = re.search(r'(?:spre|către|catre|în|in|la)\s+([a-zăâîșşţț \'.-]+)\??$', q_l)
    country = m_c.group(1).strip() if m_c else None
    if country:
        c_norm = strip_diacritics(country).lower()
        c_norm = c_norm.upper()
    else:
        c_norm = None
    return weight, c_norm

def expand_query(q: str) -> str:
    base = strip_diacritics((q or "").lower())
    extra = []
    if "ems" in base:
        extra += ["express mail service", "ems international", "serviciu express"]
    if any(k in base for k in ["pret", "preț", "cost", "tarif", "kg", "kilograme"]):
        extra += ["pret", "tarif", "cost", "greutate"]
    if "german" in base:
        extra += ["germania", "germany", "de"]
    return (q + " " + " ".join(extra)).strip()

def reformuleaza_intrebare(intrebare: str) -> str:
    t = expand_query(intrebare or "").strip().lower()
    return f"query: {t}"

# --- AR token (fix pentru „m-ar”)
AR_TOKEN_RE = re.compile(r'(?<![0-9A-Za-zăâîșşţț-])a\s*\.?\s*r\s*\.?(?![0-9A-Za-zăâîșşţț-])', re.IGNORECASE)
def has_ar_token(q: str) -> bool:
    t = strip_diacritics((q or "")).lower()
    return ("confirmare de primire" in t) or ("aviz de primire" in t) or bool(AR_TOKEN_RE.search(t))


# ------------------------------- Heuristici + „memorie de subiect”

TOPIC_INTERNAL_LETTER = "internal_letter"
TOPIC_INTERNAL_LETTER_AR = "internal_letter_ar"

def _st():
    return session_state.setdefault(g.session_id, {})

def set_topic(topic: str, **kwargs):
    s = _st()
    s["topic"] = topic
    if "ar" in kwargs:
        s["ar"] = bool(kwargs["ar"])

def get_topic():
    return session_state.get(g.session_id, {}).get("topic")

def get_topic_ar() -> bool:
    return bool(session_state.get(g.session_id, {}).get("ar", False))

def mentions_any(q: str, words):
    t = strip_diacritics((q or "").lower())
    return any(w in t for w in words)

def is_internal_route(q: str) -> bool:
    t = strip_diacritics((q or "").lower())
    return ((" din " in t or " de la " in t) and any(x in t for x in [" in ", " spre ", " catre ", " către ", " la "])
            and "romania" not in t and "italia" not in t and "extern" not in t and "international" not in t) or (" intern" in t)

def is_international_route(q: str) -> bool:
    t = strip_diacritics((q or "").lower())
    return (" international" in t) or (" din romania" in t or " din românia" in t)

def context_text() -> str:
    try:
        return (get_context() or "").lower()
    except Exception:
        return ""

def context_is_internal_letter_any() -> bool:
    t = strip_diacritics(context_text())
    return ("plic" in t or "scrisoare" in t or "corespondent" in t) and "pachet mic" not in t and "colet" not in t

def introduces_new_topic(q: str) -> bool:
    t = strip_diacritics((q or "").lower())
    if any(k in t for k in ["colet", "pachet mic", "extern", "international", "internațional",
                            "intracomunitar", "extracomunitar", "ems"]):
        return True
    if re.search(r'\b(in|în|spre|catre|către|la)\s+[a-zăâîșşţț\- ]+\b', t):
        return True
    return False

# --- Follow-up: greutate & calificatori
def is_weight_followup(q: str) -> bool:
    t = strip_diacritics((q or "").lower())
    return (
        bool(re.search(r'\d+\s*(g|gr|grame|kg)\b', t)) or
        bool(re.search(r'\b(peste|mai\s+(mare|mult)\s+de|pana\s+la|până\s+la|sub)\b', t)) or
        ("greutate" in t)
    )

def is_ambiguous_followup(q: str) -> bool:
    if not get_topic():
        return False
    t = strip_diacritics((q or "").lower())
    has_ref_words = any(w in t for w in [
        "acolo", "asta", "aia", "bine", "ok", "si", "și", "dar",
        "cum a fost", "cum e", "cum ramane", "cum rămâne",
        "prioritar", "neprioritar", "fara ar", "fără ar", "cu ar"
    ])
    return has_ref_words and not introduces_new_topic(q)


# ------------------------------- Corespondență internă (plicuri) — determinist

INTERNAL_TARIFFS = {
    "prioritar": [
        (0, 100, 6.50),
        (100, 500, 7.00),
        (500, 2000, 8.00),
    ],
    "neprioritar": [
        (0, 100, 5.00),
        (100, 500, 5.50),
        (500, 2000, 6.50),
    ],
}
# Suplimente (scutite de TVA)
SUPL_RECOMANDAT = 3.00
SUPL_VAL_DECL_REG = 5.00
SUPL_VAL_DECL_PROC = 0.01
SUPL_POST_RESTANT = 3.05
SUPL_SAMBATA_INREG = 5.08

def parse_letter_weight_grams(q: str):
    """Returnează (grams, qualifier) unde qualifier ∈ {'exact','over','under'}."""
    t_raw = (q or "")
    t = strip_diacritics(t_raw.lower())

    qualifier = "exact"
    if re.search(r'\b(peste|mai\s+(mare|mult)\s+de|>\s*)', t):
        qualifier = "over"
    elif re.search(r'\b(pana\s+la|până\s+la|sub|mai\s+(mic|putin)\s+de|<\s*)', t):
        qualifier = "under"

    m_g = re.search(r'(\d+[.,]?\d*)\s*(?:g|gr|grame)\b', t)
    m_kg = re.search(r'(\d+[.,]?\d*)\s*kg\b', t)

    grams = None
    if m_g:
        grams = float(m_g.group(1).replace(',', '.'))
    elif m_kg:
        grams = float(m_kg.group(1).replace(',', '.')) * 1000

    if grams is not None:
        grams = int(round(grams))
        if qualifier == "over":
            grams += 1
    return grams, qualifier

def parse_priority_type(q: str):
    t = strip_diacritics((q or "").lower())
    if "neprioritar" in t:
        return "neprioritar"
    if "prioritar" in t:
        return "prioritar"
    return None

def pick_internal_rate(grams: int, tip: str):
    for lo, hi, price in INTERNAL_TARIFFS.get(tip, []):
        if (grams > lo) and (grams <= hi):
            return price
        if grams == 0 and lo == 0:
            return price
    return None

def _bracket_label_for_grams(grams: int) -> str:
    if grams <= 100:
        return "≤100 g"
    if grams <= 500:
        return "101–500 g"
    if grams <= 2000:
        return "501–2000 g"
    return f"{grams} g"

def is_internal_letter_intent(q: str) -> bool:
    t = strip_diacritics((q or "").lower())
    has_item = any(k in t for k in ["plic", "scrisoare", "corespondenta", "carte postala"])
    has_route = (" din " in t or " de la " in t) and any(x in t for x in [" in ", " spre ", " catre ", " către ", " la "])
    not_parcel = "colet" not in t and "pachet mic" not in t and "extern" not in t and "international" not in t
    return has_item and has_route and not_parcel and (not has_ar_token(t))

def compute_internal_letter_answer(question: str) -> str:
    grams, _qual = parse_letter_weight_grams(question)
    if grams is None:
        grams = 100
    tip = parse_priority_type(question)

    t = strip_diacritics((question or "").lower())
    with_recomandat = "recomandat" in t
    val_decl = None
    m_val = re.search(r'valoare\s+declarat[ae]\s+(\d+[.,]?\d*)', t)
    if m_val:
        val_decl = float(m_val.group(1).replace(',', '.'))
    post_restant = ("post restant" in t) or ("post-restant" in t) or ("postrestant" in t)
    sambata = "sambata" in t or "sâmbăta" in t or "sarbator" in t or "sărbător" in t

    def extras_total(base_registered: bool) -> float:
        reg = bool(base_registered)
        add = 0.0
        if with_recomandat:
            add += SUPL_RECOMANDAT
            reg = True
        if val_decl is not None:
            add += SUPL_VAL_DECL_REG + SUPL_VAL_DECL_PROC * val_decl
            reg = True
        if post_restant:
            add += SUPL_POST_RESTANT
        if sambata and reg:
            add += SUPL_SAMBATA_INREG
        return add

    def fmt_ans(line_tip: str) -> str:
        base = pick_internal_rate(grams, line_tip)
        if base is None:
            return f"- {line_tip.title()}: nu am găsit tarif pentru {grams} g."
        total = base + extras_total(base_registered=("recomandat" in t or val_decl is not None))
        bracket = _bracket_label_for_grams(grams)
        if not (with_recomandat or val_decl is not None or post_restant or sambata):
            return f"- {line_tip.title()} ({bracket}): {fmt_lei(base)} lei"
        return (f"- {line_tip.title()} (pentru {grams} g): {fmt_lei(base)} lei + suplimente => **{fmt_lei(total)} lei**")

    if tip is None:
        lines = [
            "Pentru un plic intern, tarifele de bază sunt:",
            fmt_ans("neprioritar"),
            fmt_ans("prioritar"),
        ]
        return "\n".join(lines).strip()

    return (
        f"Pentru plic intern {tip}:\n"
        f"{fmt_ans(tip)}"
    )


# ------------------------------- Corespondență internă cu A.R.
# Tarife actuale (pachet: greutate + înregistrare + confirmare)
INTERNAL_TARIFFS_AR = {
    "neprioritar": [
        (0, 100, 10.68),
        (100, 500, 14.24),
        (500, 2000, 18.30),
    ],
    "prioritar": [
        (0, 100, 12.20),
        (100, 500, 16.27),
        (500, 2000, 25.42),
    ],
}
AR_SUPL_VAL_DECL_PROC = 0.01
AR_SUPL_POST_RESTANT = 3.05
AR_SUPL_SAMBATA_INREG = 5.08

def is_internal_letter_ar_intent(q: str) -> bool:
    t = strip_diacritics((q or "").lower())
    has_item = any(k in t for k in ["plic", "scrisoare", "corespondenta", "carte postala"])
    has_route = (" din " in t or " de la " in t) and any(x in t for x in [" in ", " spre ", " catre ", " către ", " la "])
    not_parcel = "colet" not in t and "pachet mic" not in t and "extern" not in t and "international" not in t
    return has_item and has_route and not_parcel and has_ar_token(t)

def pick_internal_rate_ar(grams: int, tip: str):
    for lo, hi, price in INTERNAL_TARIFFS_AR.get(tip, []):
        if (grams > lo) and (grams <= hi):
            return price
        if grams == 0 and lo == 0:
            return price
    return None

def compute_internal_letter_ar_answer(question: str) -> str:
    grams, _qual = parse_letter_weight_grams(question)
    if grams is None:
        grams = 100

    tip = parse_priority_type(question)
    t = strip_diacritics((question or "").lower())

    val_decl = None
    m_val = re.search(r'valoare\s+declarat[ae]\s+(\d+[.,]?\d*)', t)
    if m_val:
        val_decl = float(m_val.group(1).replace(',', '.'))
    post_restant = ("post restant" in t) or ("post-restant" in t) or ("postrestant" in t)
    sambata = "sambata" in t or "sâmbăta" in t or "sarbator" in t or "sărbător" in t

    def extras_total_ar() -> float:
        add = 0.0
        if val_decl is not None:
            add += AR_SUPL_VAL_DECL_PROC * val_decl
        if post_restant:
            add += AR_SUPL_POST_RESTANT
        if sambata:
            add += AR_SUPL_SAMBATA_INREG
        return add

    def fmt_ans_ar(line_tip: str) -> str:
        base = pick_internal_rate_ar(grams, line_tip)
        if base is None:
            return f"- {line_tip.title()}: nu am găsit tarif A.R. pentru {grams} g."
        extras = extras_total_ar()
        bracket = _bracket_label_for_grams(grams)
        if extras == 0.0:
            return f"- {line_tip.title()} A.R. ({bracket}): {fmt_lei(base)} lei"
        return (f"- {line_tip.title()} A.R. (pentru {grams} g): {fmt_lei(base)} lei + suplimente => **{fmt_lei(base + extras)} lei**")

    if tip is None:
        lines = [
            "Pentru un plic intern cu confirmare de primire (A.R.), tarifele sunt:",
            fmt_ans_ar("neprioritar"),
            fmt_ans_ar("prioritar"),
        ]
        return "\n".join(lines).strip()

    return (
        f"Pentru plic intern {tip} cu confirmare de primire (A.R.):\n"
        f"{fmt_ans_ar(tip)}"
    )


# ------------------------------- Colete externe >10 kg — determinist (fără EMS)

def get_over10_docs(query: str):
    cand = retriever.get_relevant_documents(query)

    def is_over10(d):
        src = (d.metadata.get('source_file') or '').lower()
        txt = (d.page_content or '').lower()
        return ('peste 10' in src) or ('peste 10 kg' in txt) or ('peste 10kg' in txt)

    over = [d for d in cand if is_over10(d)]
    if not over:
        hard_q = f"{query} peste 10 kg TVA inclus tarife colete externe"
        cand2 = vectorstore.similarity_search(hard_q, k=25)
        over = [d for d in cand2 if is_over10(d)]

    over = [d for d in over
            if 'ems' not in (d.page_content or '').lower()
            and 'ems' not in (d.metadata.get('source_file') or '').lower()]

    logger.info(f"Over10 docs: {len(over)}")
    for i, d in enumerate(over[:3]):
        logger.info(f"[over10 {i}] {d.metadata.get('source_file')}")
    return over

def try_calc_external_over10(question: str, docs):
    w, c = parse_weight_country(question)
    if not w or w <= 10 or not c or 'ems' in (question or '').lower():
        return None

    keep = []
    for d in docs:
        src = (d.metadata.get('source_file') or '').lower()
        txt = (d.page_content or '').lower()
        if ('peste 10' in src) or ('peste 10 kg' in txt) or ('peste 10kg' in txt):
            keep.append(d)
    if not keep:
        return None

    text = "\n".join(d.page_content for d in keep if d and d.page_content)

    pat = rf"{re.escape(c)}[^\n]*?Tarif\s*fix\s*colet.*?:\s*([\d.,]+)\s*lei[^\n]*?Tarif/?\s*\/?\s*kg.*?:\s*([\d.,]+)\s*lei"
    best = None
    for m in re.finditer(pat, text, flags=re.IGNORECASE | re.DOTALL):
        win = text[max(0, m.start() - 120) : m.end() + 120].lower()
        if "scutit de tva" in win:
            continue
        best = m
        break
    if not best:
        return None

    try:
        fix = float(best.group(1).replace('.', '').replace(',', '.'))
        perkg = float(best.group(2).replace('.', '').replace(',', '.'))
    except Exception:
        return None

    total = fix + w * perkg
    w_disp = int(w) if float(w).is_integer() else w
    w_disp_str = f"{w_disp}" if isinstance(w_disp, int) else fmt_lei(w_disp)

    rasp = (
        f"Pentru a trimite un colet extern de {w_disp_str} kg către {c.title()}, "
        f"tariful pentru **peste 10 kg (TVA inclus)** este:\n\n"
        f"1. Tarif fix: {fmt_lei(fix)} lei\n"
        f"2. Tarif per kg: {w_disp_str} kg × {fmt_lei(perkg)} lei/kg = {fmt_lei(w*perkg)} lei\n"
        f"3. Cost total: **{fmt_lei(total)} lei**"
    )
    return rasp


# ------------------------------- Generare răspuns RAG

def generate_answer(question, docs):
    try:
        context = get_context()
        full_question = f"{context}\n\nÎntrebare: {question}" if context else question

        with get_openai_callback() as cb:
            extraction_result = qa_chain.run(input_documents=docs, question=full_question)
            synthesis_result = synthesis_chain.run(
                extracted_info=extraction_result,
                question=full_question
            )

            input_cost, output_cost, total_cost = calculate_cost(cb.prompt_tokens, cb.completion_tokens)
            synthesis_result = (synthesis_result or "").replace("[Răspuns]", "").strip()
            return {
                "answer": re.sub(r'\n+', '\n', synthesis_result),
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost
            }
    except Exception as e:
        logger.error(f"Eroare generare răspuns: {str(e)}", exc_info=True)
        return {
            "answer": "Nu am putut procesa întrebarea. Poți reformula, te rog?",
            "prompt_tokens": 0, "completion_tokens": 0,
            "input_cost": 0, "output_cost": 0, "total_cost": 0
        }


# ------------------------------- Endpoints
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = getattr(g, "json_body", None) or request.get_json()
        question = (data.get('question') or '').strip()
        if not question:
            return jsonify({'error': 'Întrebare goală.'}), 400

        conversation_history[g.session_id].append(f"Utilizator: {question}")

        # -------- Follow-up context-aware: rămâne pe plic intern (cu/ fără A.R.)
        if not introduces_new_topic(question) and \
           (is_weight_followup(question) or is_ambiguous_followup(question)) and \
           (get_topic() in {TOPIC_INTERNAL_LETTER, TOPIC_INTERNAL_LETTER_AR} or context_is_internal_letter_any()):

            t = strip_diacritics((question or "").lower())
            ar = get_topic_ar()
            if "fara ar" in t or "fără ar" in t:
                ar = False
            if "cu ar" in t or has_ar_token(question):
                ar = True

            set_topic(TOPIC_INTERNAL_LETTER_AR if ar else TOPIC_INTERNAL_LETTER, ar=ar)
            answer = (compute_internal_letter_ar_answer(question) if ar
                      else compute_internal_letter_answer(question))

            if is_price_intent(question):
                answer = (answer.rstrip() + "\n\n" + PRICE_NOTE).strip()

            conversation_history[g.session_id].append(f"Asistent: {answer}")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            usage_data["timestamp"].append(ts); usage_data["session_id"].append(g.session_id)
            usage_data["question"].append(question); usage_data["prompt_tokens"].append(0)
            usage_data["completion_tokens"].append(0); usage_data["input_cost"].append(0.0)
            usage_data["output_cost"].append(0.0); usage_data["total_cost"].append(0.0)
            return jsonify({"answer": answer, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}})

        # === RUTE DETERMINISTE DIN FIȘIERE ===

        # (A) PACHET MIC INTERN
        if mentions_any(question, ["pachet mic"]) and is_internal_route(question):
            hdr = "Tarife Pachet Mic Intern (conform fișierului dedicat)."
            answer = answer_from_file(FILE_PACHET_MIC_INT_NEPRIO, hdr)
            if is_price_intent(question):
                answer = (answer.rstrip() + "\n\n" + PRICE_NOTE).strip()
            conversation_history[g.session_id].append(f"Asistent: {answer}")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            usage_data["timestamp"].append(ts); usage_data["session_id"].append(g.session_id)
            usage_data["question"].append(question); usage_data["prompt_tokens"].append(0)
            usage_data["completion_tokens"].append(0); usage_data["input_cost"].append(0.0)
            usage_data["output_cost"].append(0.0); usage_data["total_cost"].append(0.0)
            return jsonify({"answer": answer, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}})

        # (B) CORESPONDENȚĂ INTERNAȚIONALĂ fără A.R.
        if is_international_route(question) and mentions_any(question, ["plic", "scrisoare", "trimitere"]) and (not has_ar_token(question)):
            hdr = "Corespondență internațională — tarife de bază (din fișier)."
            answer = answer_from_file(FILE_CORESC_INTL, hdr)
            if is_price_intent(question):
                answer = (answer.rstrip() + "\n\n" + PRICE_NOTE).strip()
            conversation_history[g.session_id].append(f"Asistent: {answer}")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            usage_data["timestamp"].append(ts); usage_data["session_id"].append(g.session_id)
            usage_data["question"].append(question); usage_data["prompt_tokens"].append(0)
            usage_data["completion_tokens"].append(0); usage_data["input_cost"].append(0.0)
            usage_data["output_cost"].append(0.0); usage_data["total_cost"].append(0.0)
            return jsonify({"answer": answer, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}})

        # (C) CORESPONDENȚĂ INTERNAȚIONALĂ CU A.R.
        if is_international_route(question) and mentions_any(question, ["plic", "scrisoare", "trimitere"]) and has_ar_token(question):
            hdr = "Corespondență internațională cu confirmare de primire (A.R.) — tarife (din fișier)."
            answer = answer_from_file(FILE_CORESC_INTL_AR, hdr)
            if is_price_intent(question):
                answer = (answer.rstrip() + "\n\n" + PRICE_NOTE).strip()
            conversation_history[g.session_id].append(f"Asistent: {answer}")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            usage_data["timestamp"].append(ts); usage_data["session_id"].append(g.session_id)
            usage_data["question"].append(question); usage_data["prompt_tokens"].append(0)
            usage_data["completion_tokens"].append(0); usage_data["input_cost"].append(0.0)
            usage_data["output_cost"].append(0.0); usage_data["total_cost"].append(0.0)
            return jsonify({"answer": answer, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}})

        # (D) PACHET MIC INTRACOMUNITAR CU A.R.
        if mentions_any(question, ["pachet mic", "intracomunitar"]) and has_ar_token(question):
            hdr = "Pachet mic internațional intracomunitar cu A.R. — tarife (din fișier)."
            answer = answer_from_file(FILE_PACHET_MIC_INTRACOM_AR, hdr)
            if is_price_intent(question):
                answer = (answer.rstrip() + "\n\n" + PRICE_NOTE).strip()
            conversation_history[g.session_id].append(f"Asistent: {answer}")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            usage_data["timestamp"].append(ts); usage_data["session_id"].append(g.session_id)
            usage_data["question"].append(question); usage_data["prompt_tokens"].append(0)
            usage_data["completion_tokens"].append(0); usage_data["input_cost"].append(0.0)
            usage_data["output_cost"].append(0.0); usage_data["total_cost"].append(0.0)
            return jsonify({"answer": answer, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}})

        # (E) PACHET MIC EXTRACOMUNITAR CU A.R.
        if mentions_any(question, ["pachet mic", "extracomunitar"]) and has_ar_token(question):
            hdr = "Pachet mic extracomunitar cu A.R. — tarife (din fișier)."
            answer = answer_from_file(FILE_PACHET_MIC_EXTRACOM_AR, hdr)
            if is_price_intent(question):
                answer = (answer.rstrip() + "\n\n" + PRICE_NOTE).strip()
            conversation_history[g.session_id].append(f"Asistent: {answer}")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            usage_data["timestamp"].append(ts); usage_data["session_id"].append(g.session_id)
            usage_data["question"].append(question); usage_data["prompt_tokens"].append(0)
            usage_data["completion_tokens"].append(0); usage_data["input_cost"].append(0.0)
            usage_data["output_cost"].append(0.0); usage_data["total_cost"].append(0.0)
            return jsonify({"answer": answer, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}})

        # === RUTE deterministe existente

        # Plic intern cu A.R.
        if is_internal_letter_ar_intent(question):
            set_topic(TOPIC_INTERNAL_LETTER_AR, ar=True)
            answer = compute_internal_letter_ar_answer(question)
            if is_price_intent(question):
                answer = (answer.rstrip() + "\n\n" + PRICE_NOTE).strip()
            conversation_history[g.session_id].append(f"Asistent: {answer}")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            usage_data["timestamp"].append(ts); usage_data["session_id"].append(g.session_id)
            usage_data["question"].append(question); usage_data["prompt_tokens"].append(0)
            usage_data["completion_tokens"].append(0); usage_data["input_cost"].append(0.0)
            usage_data["output_cost"].append(0.0); usage_data["total_cost"].append(0.0)
            return jsonify({"answer": answer, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}})

        # Plic intern fără A.R.
        if is_internal_letter_intent(question):
            set_topic(TOPIC_INTERNAL_LETTER, ar=False)
            answer = compute_internal_letter_answer(question)
            if is_price_intent(question):
                answer = (answer.rstrip() + "\n\n" + PRICE_NOTE).strip()
            conversation_history[g.session_id].append(f"Asistent: {answer}")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            usage_data["timestamp"].append(ts); usage_data["session_id"].append(g.session_id)
            usage_data["question"].append(question); usage_data["prompt_tokens"].append(0)
            usage_data["completion_tokens"].append(0); usage_data["input_cost"].append(0.0)
            usage_data["output_cost"].append(0.0); usage_data["total_cost"].append(0.0)
            return jsonify({"answer": answer, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}})

        # Extern >10 kg (fără EMS)
        w, c = parse_weight_country(question)
        if w and w > 10 and c and 'ems' not in question.lower():
            qry = f"tarife colete externe peste 10 kg {c or ''} TVA inclus"
            docs_over10 = get_over10_docs(qry)
            special_answer = try_calc_external_over10(question, docs_over10)
            if special_answer:
                answer = special_answer.rstrip()
                if is_price_intent(question):
                    answer += "\n\n" + PRICE_NOTE
                conversation_history[g.session_id].append(f"Asistent: {answer}")
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                usage_data["timestamp"].append(ts); usage_data["session_id"].append(g.session_id)
                usage_data["question"].append(question); usage_data["prompt_tokens"].append(0)
                usage_data["completion_tokens"].append(0); usage_data["input_cost"].append(0.0)
                usage_data["output_cost"].append(0.0); usage_data["total_cost"].append(0.0)
                return jsonify({"answer": answer, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}})

        # ------------------ Flux standard (RAG)
        reformulated = reformuleaza_intrebare(question)
        docs = retriever.get_relevant_documents(reformulated)
        if not docs:
            docs = vectorstore.similarity_search(reformulated, k=10)

        logger.info(f"Docs folosite: {len(docs)}")
        for i, d in enumerate(docs[:3]):
            logger.info(f"[{i}] src={d.metadata.get('source_file')}")

        result = generate_answer(reformulated, docs)
        answer = result["answer"]

        if is_price_intent(question):
            answer = (answer.rstrip() + "\n\n" + PRICE_NOTE).strip()

        conversation_history[g.session_id].append(f"Asistent: {answer}")
        answer = answer.replace("• ", "- ").strip()

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        usage_data["timestamp"].append(ts)
        usage_data["session_id"].append(g.session_id)
        usage_data["question"].append(question)
        usage_data["prompt_tokens"].append(result["prompt_tokens"])
        usage_data["completion_tokens"].append(result["completion_tokens"])
        usage_data["input_cost"].append(result["input_cost"])
        usage_data["output_cost"].append(result["output_cost"])
        usage_data["total_cost"].append(result["total_cost"])

        if len(usage_data["timestamp"]) >= 5:
            save_to_excel()

        return jsonify({
            "answer": answer,
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_cost": result["total_cost"]
            }
        })
    except Exception as e:
        logger.error(f"Eroare în /chat: {str(e)}", exc_info=True)
        return jsonify({"error": "Eroare la procesarea întrebării."}), 500


@app.route('/save_costs', methods=['POST'])
def save_costs():
    try:
        save_to_excel()
        return jsonify({"status": "success", "message": "Costs saved to Excel"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        if usage_data:
            save_to_excel()

