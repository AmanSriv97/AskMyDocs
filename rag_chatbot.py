import streamlit as st

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
# from langchain.chains import RetrievalQA



load_dotenv(override=True)

google_api_key = os.getenv('GOOGLE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

if google_api_key:
    print(f"Google API Key exists and begins with {google_api_key[:2]}")
else:
    print("Google API Key not set (and this is optional)")

gemini = OpenAI(api_key=google_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
model_name = 'gemini-2.5-pro'#"gemini-2.0-flash"

# Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # You can tune this
    chunk_overlap=200     # Helps preserve context
)

#Create an embeddings object
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)



###################################################################################################################3
# Set the page layout
st.set_page_config(page_title=" Chatbot", page_icon="💬", layout="centered")

st.title("💬 RAG based Chatbot Interface")

with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_files = st.file_uploader("Upload a PDF or txt file",accept_multiple_files=True, type=["pdf","txt"])
    submit_file = st.button("Submit File")


extracted_texts=[]
all_chunks = []  # To hold all chunks across documents

if uploaded_files and submit_file:
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.name.split(".")[-1].lower()

        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        # Split text into chunks
        chunks = text_splitter.split_text(text)

        all_chunks.extend(chunks)  # Collect for future embedding or search
        extracted_texts.append(text)

        

    # Optional: store in session state for later use
    st.session_state.extracted_text = "\n\n---\n\n".join(extracted_texts)
    st.session_state.chunked_texts = all_chunks
    st.session_state.submitted = True
    st.session_state.uploaded_files = uploaded_files

    st.sidebar.success("✅ PDF uploaded and processed!")


    ## Create a vector store for the data
    vectorstore = FAISS.from_texts(all_chunks, embedding=embeddings)
    st.session_state.vectorstore = vectorstore




# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input from user
user_input = st.chat_input("Say something...")

if user_input:
    # Store user message

    query_embedding = embeddings.embed_query(user_input)

    # Create the retriever from the FAISS vector store
    retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # docs = vectorstore.similarity_search(query_embedding, k=3)
    docs = retriever.get_relevant_documents(user_input)

    context = "\n\n".join([doc.page_content for doc in docs])


    sys_prompt = f"""
                "You are an intelligent document assistant designed to answer questions using a retrieval-augmented generation (RAG) system. 
                Your primary goal is to provide accurate, helpful, and well-explained answers strictly based on the content retrieved from a set of 
                indexed documents. also keep normal human level understandingf of the input prompt
                The system's history information is passed here -- {st.session_state.messages}
                """

    # st.session_state.messages.append({"role": "system", "content": sys_prompt})    
    messages = [{"role": "system", "content": sys_prompt},
                {"role":"user", "content": user_input + context}]
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response (LLM response here)
    with st.spinner("Generating response..."):
        response = gemini.chat.completions.create(model=model_name, messages=messages)
        bot_response = response.choices[0].message.content
        

    # Store bot message
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
