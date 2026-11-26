import hashlib
import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pypdf import PdfReader

st.set_page_config(
    page_title="MedPolicy AI — مساعد وكلاء التأمين الصحي",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 MedPolicy AI — مساعد وكلاء التأمين الصحي")
st.caption(
    "ارفع بوالص التأمين الصحي وجداول التغطية (PDF, TXT, CSV) واحصل على إجابات فورية مدعومة برقم الصفحة والمقطع أثناء مكالمة العميل."
)


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
                    pages_text.append(f"[وثيقة: {file.name} | صفحة {idx + 1}]\n{txt}")
            documents.append("\n\n".join(pages_text))
        else:
            raw = file.getvalue()
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                content = raw.decode("latin-1", errors="ignore")
            documents.append(f"[وثيقة: {file.name}]\n{content}")
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


if "insurance_query" not in st.session_state:
    st.session_state["insurance_query"] = ""

default_key = get_default_api_key()

with st.sidebar:
    st.header("⚙️ إعدادات الوكيل")
    if default_key:
        st.success("✅ مفتاح الذكاء الاصطناعي مفعّل من إعدادات النظام")
        api_key = default_key
    else:
        api_key = st.text_input(
            "Gemini API Key (للإجابة التوليدية الفورية)",
            type="password",
            value="",
            help="أدخل مفتاح Gemini API أو احفظه في Streamlit Secrets لإخفائه عن العملاء.",
        )

    st.divider()
    st.subheader("ضبط دقة البحث في البوالص")
    chunk_size = st.slider("حجم المقطع (Chunk size)", 400, 1500, 900, 100)
    chunk_overlap = st.slider("التداخل بين المقاطع (Overlap)", 50, 300, 150, 25)
    top_k = st.slider("عدد البنود المسترجعة (Top K)", 1, 8, 4)

    st.divider()
    st.info(
        "💡 **نصيحة للوكيل:** ارفع ملف جدول المنافع (Schedule of Benefits) مع شروط البوليصة الكاملة للحصول على أدق إجابة."
    )

col_upload, col_sample = st.columns([1.2, 1])

with col_upload:
    uploaded_files = st.file_uploader(
        "📂 ارفع ملفات بوالص التأمين وجداول التغطية (PDF, TXT, CSV)",
        type=["txt", "csv", "pdf"],
        accept_multiple_files=True,
    )

with col_sample:
    sample_text = st.text_area(
        "📝 أو الصق نص بند تأميني / جدول تغطية للاختبار السريع:",
        height=135,
        placeholder="مثال: خطة التأمين الذهبية - الحد السنوي الأقصى 500,000 - نسبة التحمل للعيادات الخارجية 20% بحد أقصى 50 للزيارة - العلاج الفيزيائي مغطى حتى 12 جلسة سنوياً...",
    )

combined_text = ""
if uploaded_files:
    combined_text += extract_text_from_uploads(uploaded_files)
if sample_text.strip():
    combined_text += ("\n\n" if combined_text else "") + f"[نص بوليصة مباشر]\n{sample_text.strip()}"

if combined_text.strip():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(combined_text)
    content_hash = hashlib.sha256(
        f"{chunk_size}:{chunk_overlap}:{combined_text}".encode("utf-8", errors="ignore")
    ).hexdigest()

    with st.spinner("جاري فهرسة بنود البوليصة وتجهيز البحث الذكي..."):
        vectorstore = build_vectorstore(content_hash, tuple(chunks))

    st.success(f"✅ البوالص جاهزة للاستعلام — تم فهرسة {len(chunks)} بنداً/مقطعاً (مخزّنة في الذاكرة لسرعة الاستجابة)")

    st.markdown("### ⚡ أسئلة سريعة شائعة لوكلاء التأمين (اضغط للاختبار الفوري):")
    preset_questions = [
        "ما هي نسبة التحمل (Copay / Deductible) والحد السنوي الأقصى للتغطية؟",
        "هل العلاج الفيزيائي أو الأسنان أو النظارات مشمول في التغطية وما شروطه؟",
        "ما هي الاستثناءات الرئيسية وفترة الانتظار للأمراض المزمنة أو الحمل؟",
        "What are the deductible, out-of-pocket maximum, and exclusions in this policy?",
    ]

    btn_cols = st.columns(2)
    for idx, q_text in enumerate(preset_questions):
        with btn_cols[idx % 2]:
            if st.button(q_text, key=f"preset_{idx}", use_container_width=True):
                st.session_state["insurance_query"] = q_text

    query = st.text_input(
        "🔍 اكتب سؤال العميل حول البوليصة (بالعربية أو الإنجليزية):",
        value=st.session_state["insurance_query"],
        placeholder="مثال: هل تغطي البوليصة العمليات الجراحية الطارئة خارج الشبكة؟",
    )

    if query.strip():
        st.session_state["insurance_query"] = query
        results = vectorstore.similarity_search(query, k=top_k)

        if api_key.strip():
            try:
                context_block = "\n\n---\n\n".join([doc.page_content for doc in results])
                prompt = (
                    "أنت مساعد خبير لوكلاء التأمين الصحي. أجب بدقة ومهنية عالية بالاعتماد حصراً على بنود البوليصة المرفقة. "
                    "اذكر الأرقام والنسب والحدود المالية بوضوح، واذكر اسم الوثيقة ورقم الصفحة لكل معلومة. "
                    "إذا لم يرد ذكر الحكم في النص المرفق فاذكر ذلك صراحة ولا تخمّن.\n\n"
                    f"بنود البوليصة المسترجعة:\n{context_block}\n\n"
                    f"سؤال الوكيل/العميل:\n{query}"
                )
                answer_text, used_model = generate_with_fallback(api_key, prompt)
                st.subheader("📋 الإجابة الجاهزة للوكيل")
                st.info(answer_text)
                st.caption(f"تم التوليد عبر: {used_model} | مستند إلى {len(results)} بنود من البوليصة")
            except Exception as exc:
                st.warning(f"تعذر توليد الصياغة التلقائية ({exc}). راجع البنود المسترجعة أدناه:")

        st.subheader("📑 البنود والصفحات المرجعية من البوليصة")
        for i, doc in enumerate(results, start=1):
            with st.expander(f"مرجع #{i} — للتحقق السريع أثناء المكالمة", expanded=(i == 1)):
                st.write(doc.page_content)
else:
    st.info("👆 ارفع ملف بوليصة تأمين (PDF) أو الصق نص جدول المنافع أعلاه لبدء البحث الفوري.")
