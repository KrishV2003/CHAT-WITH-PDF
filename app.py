# Streamlit & PDF handling
import streamlit as st
from PyPDF2 import PdfReader

# LangChain Community imports (updated)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

# HuggingFace Transformers for local LLM pipeline and summarization
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Utilities
import os
from dotenv import load_dotenv
import io
import requests
from bs4 import BeautifulSoup
import speech_recognition as sr
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress HF Hub symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Streamlit page config
st.set_page_config(page_title="Chat with PDF", layout="wide")


def get_pdf_text(pdf_docs):
    """Extract text from PDF documents."""
    texts = []
    for pdf in pdf_docs:
        text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        texts.append((pdf.name, text))
    return texts


def get_text_chunks(text, chunk_size=500):
    """Split text into manageable chunks without breaking words."""
    words = text.split()
    chunks = []
    current_chunk = ""
    
    for word in words:
        if len(current_chunk) + len(word) + 1 <= chunk_size:
            current_chunk += " " + word if current_chunk else word
        else:
            chunks.append(current_chunk)
            current_chunk = word
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def embed_chunks_parallel(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Parallel embedding
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(executor.map(lambda chunk: embeddings.embed(chunk), text_chunks))

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_safe_name(pdf_name):
    return "".join(c if c.isalnum() else "_" for c in pdf_name)


def get_vector_store(text_chunks, pdf_name):
    safe_name = get_safe_name(pdf_name)
    index_path = f"faiss_index_{safe_name}"

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(index_path):
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(index_path)

    return vector_store


def get_conversational_chain():
    """Load the conversational chain for answering questions using a local HuggingFace model."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, 
    if the answer is not in the provided context, just say "answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    # Local HuggingFace model
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        tokenizer="google/flan-t5-small",
        device=-1  # CPU
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    """Handle user input and fetch the answer."""
    pdf_docs = st.session_state.pdf_docs
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    all_docs = []
    for pdf in pdf_docs:
        safe_name = get_safe_name(pdf.name)
        index_path = f"faiss_index_{safe_name}"
        if os.path.exists(index_path):
            vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            docs = vs.similarity_search(user_question, k=5)
            all_docs.extend(docs)
        else:
            st.warning(f"No index found for {pdf.name}. Please process the PDF first.")

    chain = st.session_state.qa_chain
    response = chain.invoke({"input_documents": all_docs, "question": user_question}, return_only_outputs=True)

    keywords = [
        'I cannot answer this question from the provided context',
        'I am unable to provide an answer based on the given context',
        'This question cannot be answered from the given context.',
        'Answer is not available in the context',
        'answer is not available in the context',
        'not available in the context',
        'cannot answer',
        'unable to provide'
    ]

    if any(keyword.lower() in response['output_text'].lower() for keyword in keywords):
        answer_with_resources = "The answer to your question is not available in the provided context."
    else:
        answer_with_resources = f"{response['output_text']}"

    return answer_with_resources


def summarize_text(text, tokenizer=None, model=None):
    """Summarize the provided text using T5-small."""
    try:
        if tokenizer is None:
            tokenizer = T5Tokenizer.from_pretrained("t5-small")
        if model is None:
            model = T5ForConditionalGeneration.from_pretrained("t5-small")
        
        summary = ""
        text_chunks = get_text_chunks(text, chunk_size=500)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(summarize_chunk, chunk, tokenizer, model) for chunk in text_chunks]
            for future in as_completed(futures):
                summary += "- " + future.result().capitalize() + "\n"
        
        return summary
    except Exception as e:
        st.error(f"Error occurred during summarization: {e}")
        return None


def summarize_chunk(chunk, tokenizer, model):
    """Summarize a text chunk using T5-small."""
    inputs = tokenizer.encode(
        "summarize: " + chunk,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    summary_ids = model.generate(
        inputs,
        max_length=150,
        min_length=30,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def main():
    """Main function to run the Streamlit app."""
    
    # Display image at the top
    st.image(
        "images/ai_chatbot_image_with_background-removebg-preview.png", 
        caption="HELLO!!, HOW CAN I HELP YOU TODAY?", 
        width=500
    )

    st.header("Chat with PDF 💁")
    st.markdown("Upload your PDFs, ask questions, and get answers!")

    if 'pdf_docs' not in st.session_state:
        st.session_state.pdf_docs = []
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = get_conversational_chain()

    with st.form(key='question_form'):
        user_question = st.text_input("Ask a Question from the PDF Files", value=st.session_state.user_question)
        submit_button = st.form_submit_button(label='Enter')

    if submit_button:
        answer = user_input(user_question)
        st.markdown("### Answer:")
        st.markdown(answer, unsafe_allow_html=True)

    # Sidebar PDF uploader
    st.sidebar.title("Menu:")
    pdf_docs = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True, key="pdf_uploader")

    if pdf_docs and st.sidebar.button("Process PDFs"):
        with st.spinner("Processing..."):
            st.session_state.pdf_docs = pdf_docs
            st.session_state.vector_stores = {}
            pdf_texts = get_pdf_text(pdf_docs)
            for pdf_name, raw_text in pdf_texts:
                text_chunks = get_text_chunks(raw_text)
                vs = get_vector_store(text_chunks, pdf_name)
                st.session_state.vector_stores[pdf_name] = vs
            st.success("Finished processing PDFs.")

    # Summarize PDFs
    if pdf_docs:
        selected_pdf = st.sidebar.selectbox("Select PDF to Summarize:", [pdf.name for pdf in pdf_docs])
        if st.sidebar.button("Summarize Selected PDF"):
            with st.spinner("Summarizing..."):
                pdf_texts = get_pdf_text(pdf_docs)
                for pdf_name, raw_text in pdf_texts:
                    if pdf_name == selected_pdf:
                        summary = summarize_text(raw_text)
                        st.subheader(f"Summary for {selected_pdf}:")
                        st.write(summary)
                        break

    # Microphone input (Google Speech Recognition)
    r = sr.Recognizer()
    with sr.Microphone() as source:
        if 'listening' not in st.session_state:
            st.session_state.listening = False

        button_label = "🎤 Start Listening" if not st.session_state.listening else "🎤 Stop Listening"
        if st.button(button_label):
            st.session_state.listening = not st.session_state.listening

        if st.session_state.listening:
            audio = r.record(source, duration=5)
            try:
                text = r.recognize_google(audio)
                st.session_state.user_question = text
                st.session_state.listening = False
                st.rerun()
            except sr.UnknownValueError:
                st.error("Could not understand audio")
            except sr.RequestError as e:
                st.error(f"Could not request results from Google Speech Recognition service; {e}")


if __name__ == "__main__":
    main()
