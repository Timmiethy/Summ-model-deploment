import streamlit as st
import os
import re
import time
import torch
import docx
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Các thư viện cho chức năng Hỏi-Đáp (RAG)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ==============================================================================
# PHẦN CẤU HÌNH VÀ TẢI MODEL
# ==============================================================================

# THAY ĐỔI: Hãy điền đúng ID model của bạn trên Hugging Face Hub
SUMMARIZER_MODEL_PATH = 'Timmiethy/t5-legal-summarizer-final' 
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
LLM_MODEL_NAME = "gemini-1.5-flash-latest"

# Lấy Google API Key từ Streamlit Secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
except (KeyError, FileNotFoundError):
    st.error("LỖI: Không tìm thấy GOOGLE_API_KEY. Vui lòng thiết lập trong Streamlit Secrets.")
    GOOGLE_API_KEY = None

@st.cache_resource
def load_summarizer_model():
    """Tải tokenizer và model T5 trực tiếp từ Hugging Face."""
    st.info("Đang tải mô hình Tóm tắt văn bản... Vui lòng chờ.")
    try:
        tokenizer = T5Tokenizer.from_pretrained(SUMMARIZER_MODEL_PATH)
        model = T5ForConditionalGeneration.from_pretrained(SUMMARIZER_MODEL_PATH)
        st.success("Tải mô hình Tóm tắt thành công!")
        return tokenizer, model # Trả về cả tokenizer và model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình tóm tắt: {e}")
        return None, None

@st.cache_resource
def load_embedding_model():
    """Tải mô hình embedding cho RAG."""
    st.info("Đang tải mô hình Embedding cho Hỏi-Đáp... Vui lòng chờ.")
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        st.success("Tải mô hình Embedding thành công!")
        return embeddings
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình embedding: {e}")
        return None

# ==============================================================================
# CÁC HÀM LOGIC
# ==============================================================================

def extract_key_sections(full_text: str) -> str:
    """Hàm trích lọc nội dung chính của bản án."""
    start_patterns = [
        r"NỘI DUNG VỤ ÁN\s*:?", r"NHẬN THẤY\s*:?", r"XÉT THẤY\s*:?",
        r"NHẬN ĐỊNH CỦA TÒA ÁN\s*:?", r"NHẬN ĐỊNH CỦA HỘI ĐỒNG XÉT XỬ\s*:?",
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

def summarize_text(tokenizer, model, clean_text):
    """Hàm gọi model để tóm tắt, sử dụng thư viện transformers."""
    if not clean_text.strip():
        return "Văn bản trống, không có gì để tóm tắt."
    
    final_text = re.sub(r"\s+", " ", clean_text).strip()
    prefixed_text = "summarize: " + final_text
    
    with st.spinner('Mô hình đang tóm tắt...'):
        input_ids = tokenizer.encode(prefixed_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(input_ids, max_length=256, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
    return summary

def setup_qa_chain_from_text(documents_text, embeddings_model):
    """Hàm xây dựng hệ thống QA từ danh sách các đoạn văn bản."""
    if not GOOGLE_API_KEY:
        st.error("Không thể thiết lập chuỗi QA vì thiếu Google API Key.")
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.create_documents(documents_text)
    db = FAISS.from_documents(texts, embeddings_model)
    prompt_template = """
    Bạn là một trợ lý pháp lý AI chính xác. Chỉ sử dụng thông tin trong 'Ngữ cảnh' dưới đây để trả lời câu hỏi.
    Tuyệt đối không bịa đặt hoặc dùng kiến thức ngoài.
    Nếu không tìm thấy thông tin, hãy trả lời: "Tôi không tìm thấy thông tin về vấn đề này trong các tài liệu được cung cấp."

    Ngữ cảnh: {context}
    Câu hỏi: {question}
    Trả lời (bằng tiếng Việt):
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.2)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=db.as_ retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": PROMPT}, return_source_documents=True
    )
    return qa_chain

# ==============================================================================
# GIAO DIỆN WEB STREAMLIT
# ==============================================================================

st.set_page_config(page_title="Trợ Lý Pháp Lý AI", layout="wide")
st.title("⚖️ Trợ Lý Pháp Lý AI")
st.write("Cung cấp bởi **Timmiethy** - Ứng dụng AI hỗ trợ tóm tắt và hỏi đáp văn bản pháp lý.")

# Tải các model
tokenizer, summarizer_model = load_summarizer_model()
embeddings = load_embedding_model()

tab1, tab2 = st.tabs(["📝 Tóm Tắt Văn Bản", "💬 Hỏi-Đáp Pháp Lý (RAG)"])

# --- Tab 1: Tóm tắt văn bản ---
with tab1:
    st.header("Tóm Tắt Bản Án Tự Động")
    st.write("Tải lên một file bản án (.docx) để được tiền xử lý và tóm tắt tự động.")

    uploaded_file_summary = st.file_uploader("Tải lên file .docx của bạn", type=['docx'])
    
    if uploaded_file_summary is not None:
        try:
            st.info("Đang đọc file...")
            doc = docx.Document(uploaded_file_summary)
            raw_text = "\n".join([para.text for para in doc.paragraphs])
            st.success("Đọc file thành công!")

            st.info("Đang tiền xử lý văn bản...")
            processed_text = extract_key_sections(raw_text)
            st.success("Tiền xử lý hoàn tất!")

            st.text_area("Nội dung đã được trích lọc:", value=processed_text, height=250)
            
            if st.button("Tạo Tóm Tắt"):
                if summarizer_model and tokenizer:
                    summary_result = summarize_text(tokenizer, summarizer_model, processed_text)
                    st.subheader("Bản tóm tắt:")
                    st.write(summary_result)
                else:
                    st.error("Mô hình tóm tắt chưa được tải thành công. Vui lòng kiểm tra lại logs.")
        except Exception as e:
            st.error(f"Đã có lỗi xảy ra: {e}")
    else:
        st.info("Vui lòng tải lên một file .docx để bắt đầu.")

# --- Tab 2: Hỏi-đáp ---
with tab2:
    st.header("Hỏi-Đáp Dựa Trên Tài Liệu")
    st.write("Tải lên một hoặc nhiều file văn bản (.txt) để làm cơ sở kiến thức, sau đó đặt câu hỏi.")

    uploaded_files_qa = st.file_uploader("Tải lên các file .txt của bạn", type="txt", accept_multiple_files=True)

    if uploaded_files_qa:
        documents_content = [file.read().decode("utf-8") for file in uploaded_files_qa]
        st.success(f"Đã tải lên và xử lý {len(uploaded_files_qa)} file.")
        
        # Sử dụng session_state để lưu trữ chuỗi QA, tránh build lại mỗi lần tương tác
        if 'qa_chain' not in st.session_state or st.session_state.get('last_uploaded_files') != [f.name for f in uploaded_files_qa]:
            with st.spinner("Đang xây dựng cơ sở tri thức từ tài liệu..."):
                st.session_state.qa_chain = setup_qa_chain_from_text(documents_content, embeddings)
                st.session_state.last_uploaded_files = [f.name for f in uploaded_files_qa]

        if 'qa_chain' in st.session_state and st.session_state.qa_chain:
            st.info("Hệ thống đã sẵn sàng. Hãy đặt câu hỏi của bạn.")
            question = st.text_input("Câu hỏi của bạn:")

            if question:
                with st.spinner("Đang tìm kiếm câu trả lời..."):
                    result = st.session_state.qa_chain.invoke({"query": question})
                    st.subheader("✅ Câu trả lời:")
                    st.write(result["result"])
                    with st.expander("🔍 Xem các nguồn trích dẫn"):
                        for doc in result["source_documents"]:
                            st.write(f"**Trích từ:** ...{doc.page_content[100:400].strip()}...")
