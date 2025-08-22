# app.py

import streamlit as st
import os
import re
import time
import torch

# --- Cài đặt các thư viện cần thiết ---
# Người dùng sẽ cần cài đặt chúng thông qua requirements.txt
from simpletransformers.t5 import T5Model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ==============================================================================
# PHẦN CẤU HÌNH VÀ TẢI MODEL (QUAN TRỌNG: CACHING)
# ==============================================================================

# --- Đường dẫn tới các model đã huấn luyện ---
# LƯU Ý: Khi deploy, bạn cần đảm bảo các file model này có sẵn trên server.
# Cách tốt nhất là tải chúng lên Hugging Face Hub và tải về từ đó.
# Ở đây, ta giả định chúng nằm trong thư mục 'models'.
SUMMARIZER_MODEL_PATH = "Timmiethy/t5-legal-summarizer-final"
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
LLM_MODEL_NAME = "gemini-1.5-flash-latest"

# Lấy Google API Key từ Streamlit Secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except KeyError:
    st.error(
        "LỖI: Không tìm thấy GOOGLE_API_KEY. Vui lòng thiết lập trong Streamlit Secrets."
    )
    GOOGLE_API_KEY = None  # Gán là None để tránh lỗi khi chạy


# --- Sử dụng cache của Streamlit để không phải tải lại model mỗi khi người dùng tương tác ---
@st.cache_resource
def load_summarizer_model():
    """Tải mô hình tóm tắt T5."""
    st.info("Đang tải mô hình Tóm tắt văn bản... Vui lòng chờ.")
    if not os.path.exists(SUMMARIZER_MODEL_PATH):
        return None
    try:
        model = T5Model("t5", SUMMARIZER_MODEL_PATH, use_cuda=torch.cuda.is_available())
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình tóm tắt: {e}")
        return None


@st.cache_resource
def load_embedding_model():
    """Tải mô hình embedding cho RAG."""
    st.info("Đang tải mô hình Embedding cho Hỏi-Đáp... Vui lòng chờ.")
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        return embeddings
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình embedding: {e}")
        return None


# ==============================================================================
# CÁC HÀM LOGIC TỪ CODE CỦA BẠN
# ==============================================================================


def extract_key_sections(full_text: str) -> str:
    """Hàm trích lọc nội dung chính của bản án."""
    start_patterns = [
        r"NỘI DUNG VỤ ÁN\s*:?",
        r"NHẬN THẤY\s*:?",
        r"XÉT THẤY\s*:?",
        r"NHẬN ĐỊNH CỦA TÒA ÁN\s*:?",
        r"NHẬN ĐỊNH CỦA HỘI ĐỒNG XÉT XỬ\s*:?",
    ]
    start_index = -1
    for pattern in start_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            index = match.start()
            if start_index == -1 or index < start_index:
                start_index = index
    if start_index != -1:
        return full_text[start_index:]
    return full_text


def summarize_text(model, text_to_summarize):
    """Hàm gọi model để tóm tắt."""
    if not text_to_summarize.strip():
        return "Vui lòng nhập văn bản cần tóm tắt."

    # Tiền xử lý và làm sạch văn bản đầu vào
    processed_text = extract_key_sections(text_to_summarize)
    final_cleaned_text = re.sub(r"\s+", " ", processed_text).strip()

    # Thêm prefix theo yêu cầu của model T5
    prefixed_text = "summarize: " + final_cleaned_text

    with st.spinner("Mô hình đang tóm tắt..."):
        summary = model.predict([prefixed_text])
    return summary[0]


def setup_qa_chain_from_text(documents_text, embeddings_model):
    """Hàm xây dựng hệ thống QA từ danh sách các đoạn văn bản."""
    if not GOOGLE_API_KEY:
        st.error("Không thể thiết lập chuỗi QA vì thiếu Google API Key.")
        return None

    # 2. Phân mảnh văn bản
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.create_documents(documents_text)

    # 3. Xây dựng cơ sở dữ liệu vector
    db = FAISS.from_documents(texts, embeddings_model)

    # 4. Định nghĩa prompt
    prompt_template = """
    Bạn là một trợ lý pháp lý AI chính xác. Chỉ sử dụng thông tin trong 'Ngữ cảnh' dưới đây để trả lời câu hỏi.
    Tuyệt đối không bịa đặt hoặc dùng kiến thức ngoài.
    Nếu không tìm thấy thông tin, hãy trả lời: "Tôi không tìm thấy thông tin về vấn đề này trong các tài liệu được cung cấp."

    Ngữ cảnh:
    {context}

    Câu hỏi: {question}

    Trả lời (bằng tiếng Việt):
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # 5. Thiết lập LLM và chuỗi QA
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.2)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    return qa_chain


# ==============================================================================
# GIAO DIỆN WEB STREAMLIT
# ==============================================================================

st.set_page_config(page_title="Trợ Lý Pháp Lý AI", layout="wide")

st.title("⚖️ Trợ Lý Pháp Lý AI")
st.write(
    "Ứng dụng này sử dụng AI để tóm tắt bản án và trả lời các câu hỏi pháp lý dựa trên văn bản bạn cung cấp."
)

# Tải các model cần thiết
summarizer = load_summarizer_model()
embeddings = load_embedding_model()

# Tạo các tab cho từng chức năng
tab1, tab2 = st.tabs(["📝 Tóm Tắt Văn Bản", "💬 Hỏi-Đáp Pháp Lý (RAG)"])

# --- Tab 1: Tóm tắt văn bản ---
with tab1:
    st.header("Tóm Tắt Bản Án Tự Động")
    st.write("Dán toàn bộ nội dung của một bản án vào ô dưới đây để nhận bản tóm tắt.")

    input_text = st.text_area(
        "Nội dung bản án:", height=300, placeholder="Dán nội dung vào đây..."
    )

    if st.button("Tạo Tóm Tắt"):
        if summarizer:
            if input_text:
                summary_result = summarize_text(summarizer, input_text)
                st.subheader("Bản tóm tắt:")
                st.write(summary_result)
            else:
                st.warning("Vui lòng nhập nội dung bản án.")
        else:
            st.error(
                "Mô hình tóm tắt chưa được tải. Vui lòng kiểm tra lại đường dẫn và file model."
            )

# --- Tab 2: Hỏi-đáp ---
with tab2:
    st.header("Hỏi-Đáp Dựa Trên Tài Liệu")
    st.write(
        "Tải lên một hoặc nhiều file văn bản (.txt) để làm cơ sở kiến thức, sau đó đặt câu hỏi về nội dung của chúng."
    )

    uploaded_files = st.file_uploader(
        "Tải lên các file .txt của bạn", type="txt", accept_multiple_files=True
    )

    if uploaded_files:
        documents_content = []
        for file in uploaded_files:
            documents_content.append(file.read().decode("utf-8"))

        st.success(f"Đã tải lên và xử lý {len(uploaded_files)} file.")

        # Chỉ xây dựng lại hệ thống QA nếu file được tải lên thay đổi
        # Streamlit sẽ tự động cache kết quả của hàm này
        @st.cache_data
        def get_qa_chain(_docs_content):
            with st.spinner("Đang xây dựng cơ sở tri thức từ tài liệu..."):
                return setup_qa_chain_from_text(_docs_content, embeddings)

        qa_chain = get_qa_chain(tuple(documents_content))  # Dùng tuple để có thể cache

        if qa_chain:
            st.info("Hệ thống đã sẵn sàng. Hãy đặt câu hỏi của bạn vào ô bên dưới.")
            question = st.text_input("Câu hỏi của bạn:")

            if question:
                with st.spinner("Đang tìm kiếm câu trả lời..."):
                    start_time = time.time()
                    result = qa_chain.invoke({"query": question})
                    end_time = time.time()

                    st.subheader("✅ Câu trả lời:")
                    st.write(result["result"])
                    st.write(f"*(Thời gian xử lý: {end_time - start_time:.2f} giây)*")

                    with st.expander("🔍 Xem các nguồn trích dẫn"):
                        for doc in result["source_documents"]:
                            st.write(
                                f"**Trích từ:** {doc.page_content[:250].strip()}..."
                            )
