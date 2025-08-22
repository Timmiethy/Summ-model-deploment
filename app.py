# app.py (phiên bản hoàn chỉnh)

import streamlit as st
import os
import re
import time
import torch
import docx # Cần thư viện python-docx

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
# PHẦN CẤU HÌNH VÀ TẢI MODEL
# ==============================================================================

# --- Đường dẫn tới các model ---
SUMMARIZER_MODEL_PATH = 'Timmiethy/tên-model-của-bạn-trên-hub' # <-- Sửa lại tên model của bạn trên Hub
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
LLM_MODEL_NAME = "gemini-1.5-flash-latest"

# Lấy Google API Key từ Streamlit Secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
except KeyError:
    st.error("LỖI: Không tìm thấy GOOGLE_API_KEY. Vui lòng thiết lập trong Streamlit Secrets.")
    GOOGLE_API_KEY = None

# --- Cache để không tải lại model ---
@st.cache_resource
def load_summarizer_model():
    st.info("Đang tải mô hình Tóm tắt văn bản... Vui lòng chờ.")
    try:
        model = T5Model("t5", SUMMARIZER_MODEL_PATH, use_cuda=torch.cuda.is_available())
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình tóm tắt: {e}")
        return None

@st.cache_resource
def load_embedding_model():
    st.info("Đang tải mô hình Embedding cho Hỏi-Đáp... Vui lòng chờ.")
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        return embeddings
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình embedding: {e}")
        return None

# ==============================================================================
# CÁC HÀM LOGIC (BAO GỒM TIỀN XỬ LÝ)
# ==============================================================================

def extract_key_sections(full_text: str) -> str:
    """Hàm trích lọc nội dung chính của bản án (lấy từ code tiền xử lý của bạn)."""
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
        # Trả về phần văn bản đã được cắt lọc
        return full_text[start_index:]
    # Nếu không tìm thấy, trả về toàn bộ
    return full_text

# --- THAY ĐỔI QUAN TRỌNG 1 ---
# Hàm tóm tắt giờ đây chỉ nhận văn bản đã sạch, không cần xử lý lại
def summarize_text(model, clean_text):
    """Hàm gọi model để tóm tắt văn bản đã được tiền xử lý."""
    if not clean_text.strip():
        return "Vui lòng nhập văn bản cần tóm tắt."
    
    # Chỉ cần thêm prefix và chuẩn hóa khoảng trắng
    final_text = re.sub(r"\s+", " ", clean_text).strip()
    prefixed_text = "summarize: " + final_text
    
    with st.spinner('Mô hình đang tóm tắt...'):
        summary = model.predict([prefixed_text])
    return summary[0]

# (Các hàm khác cho Tab 2 giữ nguyên)
def setup_qa_chain_from_text(documents_text, embeddings_model):
    # ... (giữ nguyên code cũ)
    pass

# ==============================================================================
# GIAO DIỆN WEB STREAMLIT
# ==============================================================================

st.set_page_config(page_title="Trợ Lý Pháp Lý AI", layout="wide")
st.title("⚖️ Trợ Lý Pháp Lý AI")

# Tải các model
summarizer = load_summarizer_model()
embeddings = load_embedding_model()

tab1, tab2 = st.tabs(["📝 Tóm Tắt Văn Bản", "💬 Hỏi-Đáp Pháp Lý (RAG)"])

# --- Tab 1: Tóm tắt văn bản (đã nâng cấp) ---
with tab1:
    st.header("Tóm Tắt Bản Án Tự Động")
    st.write("Tải lên một file bản án (.docx) để được tiền xử lý và tóm tắt tự động.")

    uploaded_file_summary = st.file_uploader(
        "Tải lên file .docx của bạn",
        type=['docx'] # Chỉ chấp nhận file docx
    )
    
    # --- THAY ĐỔI QUAN TRỌNG 2 ---
    # Luồng xử lý khi người dùng tải file lên
    if uploaded_file_summary is not None:
        try:
            # 1. Đọc nội dung thô từ file
            st.info("Đang đọc file...")
            doc = docx.Document(uploaded_file_summary)
            raw_text = "\n".join([para.text for para in doc.paragraphs])
            st.success("Đọc file thành công!")

            # 2. Chạy hàm tiền xử lý
            st.info("Đang tiền xử lý văn bản...")
            processed_text = extract_key_sections(raw_text)
            st.success("Tiền xử lý hoàn tất! Văn bản đã sẵn sàng để tóm tắt.")

            # Hiển thị văn bản đã được xử lý cho người dùng xem
            st.text_area("Nội dung đã được tiền xử lý:", value=processed_text, height=250)
            
            # 3. Đưa vào mô hình
            if st.button("Tạo Tóm Tắt"):
                if summarizer:
                    summary_result = summarize_text(summarizer, processed_text)
                    st.subheader("Bản tóm tắt:")
                    st.write(summary_result)
                else:
                    st.error("Mô hình tóm tắt chưa được tải.")

        except Exception as e:
            st.error(f"Đã có lỗi xảy ra: {e}")
    else:
        st.info("Vui lòng tải lên một file .docx để bắt đầu.")

# --- Tab 2: Hỏi-đáp ---
with tab2:
    # ... (giữ nguyên code cũ)
    pass
