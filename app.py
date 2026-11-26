import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pypdf import PdfReader

st.set_page_config(page_title="RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Assistant — مساعد البحث والاسترجاع الذكي")
st.caption("ارفع ملفاتك (TXT, CSV, PDF) أو أدخل نصاً، ثم اطرح سؤالك للبحث في المحتوى.")


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


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
                    pages_text.append(f"[صفحة {idx + 1} - {file.name}]\n{txt}")
            documents.append("\n\n".join(pages_text))
        else:
            raw = file.getvalue()
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                content = raw.decode("latin-1", errors="ignore")
            documents.append(f"[ملف: {file.name}]\n{content}")
    return "\n\n".join(documents)


with st.sidebar:
    st.header("إعدادات المشروع")
    api_key = st.text_input(
        "Gemini API Key (اختياري للإجابة التوليدية)",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="يمكنك تركه فارغاً لعرض المقاطع الأكثر صلة مباشرة من ملفاتك.",
    )
    chunk_size = st.slider("حجم المقطع النصي (Chunk size)", 300, 1500, 900, 100)
    chunk_overlap = st.slider("التداخل بين المقاطع (Overlap)", 0, 300, 150, 25)
    top_k = st.slider("عدد المقاطع المسترجعة (Top K)", 1, 8, 4)

uploaded_files = st.file_uploader(
    "ارفع ملفات البيانات (TXT, CSV, PDF)",
    type=["txt", "csv", "pdf"],
    accept_multiple_files=True,
)

sample_text = st.text_area(
    "أو الصق نصاً تجريبياً هنا للبدء السريع:",
    height=140,
    placeholder="اكتب أو الصق أي نص أو ملاحظات علمية أو بيانات هنا لاختبار البحث والاسترجاع...",
)

combined_text = ""
if uploaded_files:
    combined_text += extract_text_from_uploads(uploaded_files)
if sample_text.strip():
    combined_text += ("\n\n" if combined_text else "") + f"[نص مباشر]\n{sample_text.strip()}"

if combined_text.strip():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(combined_text)

    with st.spinner("جاري فهرسة النصوص وبناء قاعدة المتجهات..."):
        embeddings = load_embeddings()
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    st.success(f"تم تجهيز الفهرس بنجاح — عدد المقاطع المفهرسة: {len(chunks)}")

    query = st.text_input("اطرح سؤالك حول البيانات المرفوعة:")
    if query.strip():
        results = vectorstore.similarity_search(query, k=top_k)

        if api_key.strip():
            try:
                from google import genai

                client = genai.Client(api_key=api_key.strip())
                context_block = "\n\n---\n\n".join([doc.page_content for doc in results])
                prompt = (
                    "أجب بدقة ووضوح اعتماداً على السياق المرفق فقط. "
                    "إذا لم تجد الإجابة في السياق فاذكر ذلك صراحة.\n\n"
                    f"السياق:\n{context_block}\n\n"
                    f"السؤال:\n{query}"
                )
                interaction = client.interactions.create(
                    model="gemini-3.8-flash",
                    input=prompt,
                )
                st.subheader("الإجابة المولّدة")
                st.write(interaction.output_text)
            except Exception as exc:
                st.warning(f"تعذر توليد إجابة عبر Gemini API ({exc}). تم عرض المقاطع المسترجعة:")

        st.subheader("المقاطع الأكثر صلة من مصادرك")
        for i, doc in enumerate(results, start=1):
            with st.expander(f"مقطع #{i}", expanded=(i == 1)):
                st.write(doc.page_content)
else:
    st.info("ارفع ملفاً (TXT / CSV / PDF) أو أدخل نصاً في الحقل أعلاه لتفعيل البحث.")
