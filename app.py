"""
app.py
------
Ye file poore RAG system ka "face" hai - Streamlit se ek simple web
interface banate hai jaha user:
1. Apna PDF upload kar sake
2. Us document ke baare mein sawaal puch sake
3. AI ka answer dekh sake

Chalane ka tarika: streamlit run app.py
"""
import streamlit as st
import os
from document_loader import extract_text_from_pdf, chunk_text
from vector_store import store_chunks
from rag_engine import get_answer

# Page ka basic setup (title, icon jo browser tab mein dikhega)
st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")

st.title("📄 RAG Document Q&A System")
st.write("Apna PDF upload karo aur uske baare mein kuch bhi puch lo!")

# ---- Session state setup ----
# Streamlit har baar user kuch bhi karta hai (jaise button click) to
# poori script dobara run hoti hai. Isliye hume "yaad rakhna" padta hai
# ki document already process ho chuka hai ya nahi - warna baar baar
# reprocess hota rahega.
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False


# ---- Step 1: PDF Upload ----
st.header("Step 1: Document Upload Karo")

uploaded_file = st.file_uploader("Apna PDF file yaha upload karo", type=["pdf"])

if uploaded_file is not None:
    # Uploaded file ko temporarily save karo (kyunki hume file path chahiye)
    temp_path = "uploaded_temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process button
    if st.button("Document Process Karo"):
        with st.spinner("PDF se text nikal rahe hai aur embeddings bana rahe hai..."):
            # Document loader use karke text nikalo aur chunks banao
            text = extract_text_from_pdf(temp_path)
            chunks = chunk_text(text)

            # Chunks ko ChromaDB mein store karo
            store_chunks(chunks)

            st.session_state.document_processed = True

        st.success(f"Document process ho gaya! {len(chunks)} chunks bane.")


# ---- Step 2: Question aur Answer ----
if st.session_state.document_processed:
    st.header("Step 2: Sawaal Pucho")

    user_question = st.text_input("Apna question yaha likho:")

    if st.button("Answer Dhundo"):
        if user_question.strip() == "":
            st.warning("Pehle koi question likho.")
        else:
            with st.spinner("Answer dhundh rahe hai..."):
                answer = get_answer(user_question)

            st.subheader("Answer:")
            st.write(answer)
else:
    st.info("Pehle upar PDF upload karke 'Document Process Karo' button dabao.")