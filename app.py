import hashlib
import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pypdf import PdfReader

st.set_page_config(
    page_title="MedPolicy AI — Health Insurance Policy Assistant",
    page_icon="🏥",
    layout="wide",
)

UI_TEXT = {
    "English": {
        "title": "🏥 MedPolicy AI — Health Insurance Policy Assistant",
        "caption": "Upload health insurance policies and benefit schedules (PDF, TXT, CSV) for instant answers with page & clause citations during live client calls.",
        "sidebar_header": "⚙️ Workspace Settings",
        "secret_ok": "✅ AI engine active via workspace secrets",
        "api_label": "Gemini API Key (for instant AI answers)",
        "api_help": "Enter your Gemini API key or store GEMINI_API_KEY in Streamlit Secrets.",
        "retrieval_header": "Retrieval Settings",
        "chunk_size": "Chunk size",
        "chunk_overlap": "Chunk overlap",
        "top_k": "Top matching clauses (Top K)",
        "upload_label": "📂 Upload insurance policies & benefit schedules (PDF, TXT, CSV)",
        "sample_label": "📝 Or paste a policy clause / benefit summary for quick testing:",
        "sample_placeholder": "Example: Gold PPO Plan — Annual deductible: $1,500 in-network / $3,000 out-of-network. Out-of-pocket max: $6,000. Primary care copay: $25. Physical therapy: covered up to 20 visits/year with prior authorization.",
        "spinner": "Indexing policy clauses and building vector store...",
        "index_ready": "✅ Policy index ready — {count} chunks cached in memory for fast lookup",
        "quick_header": "### ⚡ Quick Questions for Health Insurance Agents (Click to test):",
        "presets": [
            "What are the in-network and out-of-network deductibles and out-of-pocket maximum?",
            "Is physical therapy, urgent care, or prescription coverage included, and are there copays or prior authorizations?",
            "What are the main exclusions, waiting periods, or out-of-network coverage limits?",
            "Summarize emergency care coverage and claim filing rules from this policy.",
        ],
        "query_label": "🔍 Ask a question about the uploaded policy:",
        "query_placeholder": "Example: Does this plan cover out-of-network emergency surgery?",
        "answer_header": "📋 Agent-Ready Answer",
        "model_caption": "Model: {model} | Grounded in {k} retrieved policy excerpts",
        "fallback_warn": "Could not generate AI summary ({err}). Review the retrieved policy clauses below:",
        "sources_header": "📑 Source Excerpts & Page Citations",
        "excerpt_title": "Excerpt #{idx} — Verification Reference",
        "empty_info": "👆 Upload a policy PDF/TXT/CSV or paste a benefit excerpt above to start searching.",
        "answer_lang_instruction": "Respond in clear, professional English suitable for a licensed US health insurance agent.",
    },
    "Español": {
        "title": "🏥 MedPolicy AI — Asistente de Pólizas de Seguro Médico",
        "caption": "Suba pólizas de seguro médico y tablas de beneficios (PDF, TXT, CSV) para obtener respuestas inmediatas con citas de página y cláusula durante llamadas con clientes.",
        "sidebar_header": "⚙️ Configuración del Espacio",
        "secret_ok": "✅ Motor de IA activo desde los secretos del sistema",
        "api_label": "Clave API de Gemini (para respuestas instantáneas)",
        "api_help": "Ingrese su clave API de Gemini o guárdela en Streamlit Secrets.",
        "retrieval_header": "Ajustes de Búsqueda",
        "chunk_size": "Tamaño del fragmento (Chunk size)",
        "chunk_overlap": "Superposición (Overlap)",
        "top_k": "Cláusulas principales (Top K)",
        "upload_label": "📂 Suba pólizas de seguro y tablas de beneficios (PDF, TXT, CSV)",
        "sample_label": "📝 O pegue una cláusula / resumen de beneficios para prueba rápida:",
        "sample_placeholder": "Ejemplo: Plan Gold PPO — Deducible anual: $1,500 dentro de la red / $3,000 fuera de la red. Máximo de bolsillo: $6,000. Copago atención primaria: $25.",
        "spinner": "Indexando cláusulas de la póliza y construyendo base vectorial...",
        "index_ready": "✅ Índice listo — {count} fragmentos guardados en memoria para búsqueda rápida",
        "quick_header": "### ⚡ Preguntas Rápidas para Agentes de Seguros (Haga clic para probar):",
        "presets": [
            "¿Cuáles son los deducibles dentro y fuera de la red y el máximo de bolsillo anual?",
            "¿Están cubiertos la terapia física, atención de urgencia o medicamentos recetados, y qué copagos aplican?",
            "¿Cuáles son las principales exclusiones, períodos de espera o límites fuera de la red?",
            "Resuma la cobertura de atención de emergencia y las reglas para presentar reclamos.",
        ],
        "query_label": "🔍 Haga una pregunta sobre la póliza cargada:",
        "query_placeholder": "Ejemplo: ¿Este plan cubre cirugía de emergencia fuera de la red?",
        "answer_header": "📋 Respuesta Lista para el Agente",
        "model_caption": "Modelo: {model} | Basado en {k} fragmentos de la póliza",
        "fallback_warn": "No se pudo generar el resumen con IA ({err}). Revise las cláusulas recuperadas abajo:",
        "sources_header": "📑 Extractos de Origen y Citas de Página",
        "excerpt_title": "Extracto #{idx} — Referencia de Verificación",
        "empty_info": "👆 Suba una póliza en PDF/TXT/CSV o pegue un extracto arriba para comenzar.",
        "answer_lang_instruction": "Respond in clear, professional Spanish (Español) suitable for a licensed US health insurance agent assisting Spanish-speaking clients, while keeping exact dollar amounts, percentages, and document page citations.",
    },
}


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def build_vectorstore(content_hash: str, chunks_tuple: tuple):
    embeddings = load_embeddings()
    return FAISS.from_texts(list(chunks_tuple), embedding=embeddings)


def get_default_api_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


def extract_text_from_uploads(uploaded_files):
    documents = []
    for file in uploaded_files:
        name = file.name.lower()
        if name.endswith(".pdf"):
            reader = PdfReader(file)
            pages_text = []
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                if txt.strip():
                    pages_text.append(f"[Document: {file.name} | Page {idx + 1}]\n{txt}")
            documents.append("\n\n".join(pages_text))
        else:
            raw = file.getvalue()
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                content = raw.decode("latin-1", errors="ignore")
            documents.append(f"[Document: {file.name}]\n{content}")
    return "\n\n".join(documents)


def generate_with_fallback(api_key_value: str, prompt_text: str):
    from google import genai

    client = genai.Client(api_key=api_key_value.strip())
    candidate_models = [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
    ]
    last_error = None
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt_text,
            )
            if getattr(response, "text", None):
                return response.text, model_name
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(str(last_error))


def translate_to_arabic_helper(api_key_value: str, source_text: str):
    translation_prompt = (
        "Translate the following insurance policy answer accurately into clear, natural Arabic "
        "while preserving exact numbers, percentages, dollar amounts, and page numbers:\n\n"
        f"{source_text}"
    )
    ar_text, _ = generate_with_fallback(api_key_value, translation_prompt)
    return ar_text


if "insurance_query" not in st.session_state:
    st.session_state["insurance_query"] = ""

default_key = get_default_api_key()

with st.sidebar:
    client_lang = st.radio(
        "🌐 Client Language / Idioma del Cliente",
        options=["English", "Español"],
        horizontal=True,
    )
    t = UI_TEXT[client_lang]

    st.divider()
    st.header(t["sidebar_header"])
    if default_key:
        st.success(t["secret_ok"])
        api_key = default_key
    else:
        api_key = st.text_input(
            t["api_label"],
            type="password",
            value="",
            help=t["api_help"],
        )

    st.divider()
    st.subheader(t["retrieval_header"])
    chunk_size = st.slider(t["chunk_size"], 400, 1500, 900, 100)
    chunk_overlap = st.slider(t["chunk_overlap"], 50, 300, 150, 25)
    top_k = st.slider(t["top_k"], 1, 8, 4)

    st.divider()
    st.subheader("Owner Review Helper")
    show_arabic_helper = st.toggle(
        "Show Arabic translation for my review",
        value=False,
        help="Optional helper for you while testing or reviewing answers in Arabic.",
    )

st.title(t["title"])
st.caption(t["caption"])

col_upload, col_sample = st.columns([1.2, 1])

with col_upload:
    uploaded_files = st.file_uploader(
        t["upload_label"],
        type=["txt", "csv", "pdf"],
        accept_multiple_files=True,
    )

with col_sample:
    sample_text = st.text_area(
        t["sample_label"],
        height=135,
        placeholder=t["sample_placeholder"],
    )

combined_text = ""
if uploaded_files:
    combined_text += extract_text_from_uploads(uploaded_files)
if sample_text.strip():
    combined_text += ("\n\n" if combined_text else "") + f"[Direct Policy Excerpt]\n{sample_text.strip()}"

if combined_text.strip():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(combined_text)
    content_hash = hashlib.sha256(
        f"{chunk_size}:{chunk_overlap}:{combined_text}".encode("utf-8", errors="ignore")
    ).hexdigest()

    with st.spinner(t["spinner"]):
        vectorstore = build_vectorstore(content_hash, tuple(chunks))

    st.success(t["index_ready"].format(count=len(chunks)))

    st.markdown(t["quick_header"])
    btn_cols = st.columns(2)
    for idx, q_text in enumerate(t["presets"]):
        with btn_cols[idx % 2]:
            if st.button(q_text, key=f"preset_{client_lang}_{idx}", use_container_width=True):
                st.session_state["insurance_query"] = q_text

    query = st.text_input(
        t["query_label"],
        value=st.session_state["insurance_query"],
        placeholder=t["query_placeholder"],
    )

    if query.strip():
        st.session_state["insurance_query"] = query
        results = vectorstore.similarity_search(query, k=top_k)

        if api_key.strip():
            try:
                context_block = "\n\n---\n\n".join([doc.page_content for doc in results])
                prompt = (
                    "You are a precise health insurance policy assistant for licensed US insurance agents. "
                    f"{t['answer_lang_instruction']} "
                    "Answer strictly using the provided policy excerpts. State exact dollar amounts, percentages, "
                    "deductibles, copays, limits, and cite the document name and page number for every claim. "
                    "If the policy excerpts do not state a rule, say so explicitly without guessing.\n\n"
                    f"Policy Excerpts:\n{context_block}\n\n"
                    f"Agent Question:\n{query}"
                )
                answer_text, used_model = generate_with_fallback(api_key, prompt)
                st.subheader(t["answer_header"])
                st.info(answer_text)
                st.caption(t["model_caption"].format(model=used_model, k=len(results)))

                if show_arabic_helper:
                    with st.expander("🇸🇦 ترجمة مساعدة لك فقط (للمراجعة بالعربية)", expanded=True):
                        try:
                            ar_translation = translate_to_arabic_helper(api_key, answer_text)
                            st.write(ar_translation)
                        except Exception as tr_exc:
                            st.warning(f"Arabic helper unavailable: {tr_exc}")
            except Exception as exc:
                st.warning(t["fallback_warn"].format(err=exc))

        st.subheader(t["sources_header"])
        for i, doc in enumerate(results, start=1):
            with st.expander(t["excerpt_title"].format(idx=i), expanded=(i == 1)):
                st.write(doc.page_content)
else:
    st.info(t["empty_info"])
