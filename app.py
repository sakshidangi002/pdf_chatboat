import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import fitz
import pdfplumber
import os
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_cohere import CohereRerank


# Setup event loop for Streamlit
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

load_dotenv()
st.title(" PDF Chatbot")


# Extract text, tables, and images

def get_pdf_content(pdf_docs):
    all_content = ""

    for pdf in pdf_docs:
        # Save uploaded file temporarily
        temp_path = os.path.join("temp_" + pdf.name)
        with open(temp_path, "wb") as f:
            f.write(pdf.read())

        # 1️ --- Extract text ---
        with fitz.open(temp_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text.strip():
                    all_content += f"\n--- Page {page_num} Text ---\n{text}"

        # 2️ --- Extract tables ---
        with pdfplumber.open(temp_path) as doc:
            for page_num, page in enumerate(doc.pages, start=1):
                tables = page.extract_tables()
                for t_index, table in enumerate(tables, start=1):
                    if not table:
                        continue
                    table_str = "\n".join(
                        [", ".join(str(cell) if cell is not None else "" for cell in row)
                         for row in table if any(row)]
                    )
                    all_content += f"\n--- Page {page_num} Table {t_index} ---\n{table_str}"

        # 3️ --- Extract images ---
        with fitz.open(temp_path) as doc:
            image_folder = "pdf_images"
            os.makedirs(image_folder, exist_ok=True)
            for page_num, page in enumerate(doc, start=1):
                for img_index, img in enumerate(page.get_images(full=True), start=1):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_name = f"page_{page_num}_img_{img_index}.png"
                    with open(os.path.join(image_folder, image_name), "wb") as f:
                        f.write(image_bytes)
                    all_content += f"\n--- Page {page_num} Image {img_index} ---\n[Image saved as {image_name}]"

        os.remove(temp_path)

    return all_content



# Chunking
def get_text_chunks(all_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(all_text)


# Vector store creation
def get_vector_store(chunk_text):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.from_texts(chunk_text, embedding=embedding_model)
    db.save_local("faiss_index")



def get_hybrid_retriever(chunk_text):
    # Embedding-based vector retriever (semantic)
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = FAISS.from_texts(chunk_text, embedding=embedding_model)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 8})

    # BM25 retriever (keyword-based)
    bm25_retriever = BM25Retriever.from_texts(chunk_text)
    bm25_retriever.k = 8

    # Combine them: weighted hybrid retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]  # semantic gets more weight
    )

    return hybrid_retriever


def get_reranker():
    return CohereRerank(model="rerank-english-v3.0", top_n=5)


# QA chain
def conversation_chain():
    prompt_template = """You are an intelligent assistant that answers questions from PDF content.
    The context may include plain text, tables (in comma-separated format), or image references.
    When a table is provided, interpret it carefully and extract values to answer precisely.
    If the answer is not available in the context, say: "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = OllamaLLM(model="gemma:2b", temperature=0.1)
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Handle user queries
def user_input(user_question, chunk_text):
    st.write(" Performing Hybrid Search...")#remove

    # 1️ Create hybrid retriever
    hybrid_retriever = get_hybrid_retriever(chunk_text)

    # 2️ Retrieve top results (combined semantic + keyword)
    retrieved_docs = hybrid_retriever.invoke(user_question)

    st.write(f" Retrieved {len(retrieved_docs)} relevant chunks")

    # 3️ Rerank results using Cohere (optional)
    try:
        reranker = get_reranker()
        reranked_docs = reranker.compress_documents(retrieved_docs, user_question)
        final_docs = reranked_docs
        st.success(" Reranking complete — best chunks selected!")#remove
    except Exception as e:
        st.warning(" Reranker unavailable. Using hybrid results only.")#remove
        final_docs = retrieved_docs

    # 4️ Pass final docs to QA chain
    chain = conversation_chain()
    response = chain({"input_documents": final_docs, "question": user_question}, return_only_outputs=True)
    
    st.markdown("###  Answer:")
    st.write(response["output_text"])

    # Show top retrieved chunks
    #with st.expander(" View Retrieved Context Chunks"):
    #    for i, doc in enumerate(final_docs, start=1):
    #        st.markdown(f"**Chunk {i}:**")
    #        st.text_area(f"Chunk {i} Content", doc.page_content, height=180)


# ----------------------------------------------------------
# Streamlit App
# ----------------------------------------------------------
def main():
    st.header(" Chat with PDF — Includes Tables & Images")

    # Persistent storage for processed chunks
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = None

    user_question = st.text_input("Ask a question about your PDF:")

    with st.sidebar:
        st.write(" Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        if st.button("Process PDF"):
            if pdf_docs:
                with st.spinner(" Extracting and chunking your PDF..."):
                    raw_text = get_pdf_content(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state.text_chunks = text_chunks  #  Save chunks in session
                    st.success(" PDF processed successfully!")

                    st.subheader(" Extracted & Chunked Data Preview")
                    st.write(f"Total Chunks Created: {len(text_chunks)}")

                    with st.expander("Click to view chunks"):
                        for i, chunk in enumerate(text_chunks, start=1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text_area(f"Chunk {i} Content", chunk, height=180)
            else:
                st.warning(" Please upload a PDF first.")

    # Handle user question only if chunks exist
    if user_question:
        if st.session_state.text_chunks:
            user_input(user_question, st.session_state.text_chunks)
        else:
            st.warning(" Please upload and process a PDF before asking questions.")

if __name__ == "__main__":
    main()
