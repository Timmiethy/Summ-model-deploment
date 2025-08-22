# app.py (tiếp theo và hoàn chỉnh)

def summarize_text(tokenizer, model, clean_text):
    """Hàm gọi model để tóm tắt, sử dụng thư viện transformers."""
    if not clean_text.strip():
        return "Vui lòng nhập văn bản cần tóm tắt."
    
    # Chuẩn bị văn bản đầu vào cho model T5
    final_text = re.sub(r"\s+", " ", clean_text).strip()
    prefixed_text = "summarize: " + final_text
    
    with st.spinner('Mô hình đang tóm tắt...'):
        # 1. Mã hóa văn bản thành các ID
        input_ids = tokenizer.encode(prefixed_text, return_tensors="pt", max_length=1024, truncation=True)
        
        # 2. Tạo bản tóm tắt
        summary_ids = model.generate(input_ids, max_length=256, num_beams=4, early_stopping=True)
        
        # 3. Giải mã các ID thành văn bản
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
    return summary

def setup_qa_chain_from_text(documents_text, embeddings_model):
    """Hàm xây dựng hệ thống QA từ danh sách các đoạn văn bản."""
    # (Hàm này giữ nguyên, không cần thay đổi)
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

    Ngữ cảnh:
    {context}

    Câu hỏi: {question}

    Trả lời (bằng tiếng Việt):
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.2)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return qa_chain

# ==============================================================================
# GIAO DIỆN WEB STREAMLIT
# ==============================================================================

st.set_page_config(page_title="Trợ Lý Pháp Lý AI", layout="wide")
st.title("⚖️ Trợ Lý Pháp Lý AI")

# Tải các model
# --- THAY ĐỔI 3: Nhận về cả tokenizer và model ---
tokenizer, summarizer_model = load_summarizer_model()
embeddings = load_embedding_model()

tab1, tab2 = st.tabs(["📝 Tóm Tắt Văn Bản", "💬 Hỏi-Đáp Pháp Lý (RAG)"])

# --- Tab 1: Tóm tắt văn bản ---
with tab1:
    st.header("Tóm Tắt Bản Án Tự Động")
    st.write("Tải lên một file bản án (.docx) để được tiền xử lý và tóm tắt tự động.")

    uploaded_file_summary = st.file_uploader(
        "Tải lên file .docx của bạn",
        type=['docx']
    )
    
    if uploaded_file_summary is not None:
        try:
            st.info("Đang đọc file...")
            doc = docx.Document(uploaded_file_summary)
            raw_text = "\n".join([para.text for para in doc.paragraphs])
            st.success("Đọc file thành công!")

            st.info("Đang tiền xử lý văn bản...")
            processed_text = extract_key_sections(raw_text)
            st.success("Tiền xử lý hoàn tất! Văn bản đã sẵn sàng để tóm tắt.")

            st.text_area("Nội dung đã được tiền xử lý:", value=processed_text, height=250)
            
            if st.button("Tạo Tóm Tắt"):
                # Kiểm tra xem model và tokenizer đã được tải thành công chưa
                if summarizer_model and tokenizer:
                    # --- THAY ĐỔI 4: Truyền cả tokenizer và model vào hàm ---
                    summary_result = summarize_text(tokenizer, summarizer_model, processed_text)
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
    st.header("Hỏi-Đáp Dựa Trên Tài Liệu")
    st.write("Tải lên một hoặc nhiều file văn bản (.txt) để làm cơ sở kiến thức, sau đó đặt câu hỏi về nội dung của chúng.")

    uploaded_files = st.file_uploader(
        "Tải lên các file .txt của bạn",
        type="txt",
        accept_multiple_files=True
    )

    if uploaded_files:
        documents_content = [file.read().decode("utf-8") for file in uploaded_files]
        st.success(f"Đã tải lên và xử lý {len(uploaded_files)} file.")
        
        @st.cache_data
        def get_qa_chain(_docs_content):
            with st.spinner("Đang xây dựng cơ sở tri thức từ tài liệu..."):
                return setup_qa_chain_from_text(_docs_content, embeddings)

        qa_chain = get_qa_chain(tuple(documents_content))

        if qa_chain:
            st.info("Hệ thống đã sẵn sàng. Hãy đặt câu hỏi của bạn vào ô bên dưới.")
            question = st.text_input("Câu hỏi của bạn:")

            if question:
                with st.spinner("Đang tìm kiếm câu trả lời..."):
                    result = qa_chain.invoke({"query": question})
                    st.subheader("✅ Câu trả lời:")
                    st.write(result["result"])
