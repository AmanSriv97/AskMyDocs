# AskMyDocs (🧠 Chat with Your Documents)
AskMyDocs is an intelligent chatbot that lets you interact with your documents like never before. Powered by advanced Large Language Models (LLMs), it understands the content of your uploaded files — whether it's one document or many — and answers your questions with full contextual awareness. It's like giving your documents a brain!

A powerful, **RAG based** LLM-powered chatbot that lets you **converse with your documents**. Upload one or more files, and ask questions directly — as if you're chatting with the documents themselves.

## 🚀 Features

- 📄 **Multi-document support** – Upload **multiple documents** (PDF, TXT, DOCX, etc.) at once.
- 🧠 **LLM-Powered** – Uses cutting-edge Large Language Models to understand and answer queries contextually.
- 🔍 **Contextual understanding** – Embeds document content and provides **accurate answers** based on the uploaded files.
- 💬 **Natural chat interface** – Talk to your documents just like you would in a regular chat.
- ⚡ **Fast and responsive** – Optimized for performance and low latency responses.

## 🛠️ How It Works

1. Upload one or more documents.
2. The content is extracted and converted into vector embeddings.
3. The embeddings are stored and indexed.
4. When a question is asked, the chatbot retrieves the most relevant chunks from the database.
5. The query, along with the context, is passed to the LLM to generate a response.

## 📦 Tech Stack

- 🧠 **LLM Backend**: OpenAI / Gemini (configurable)
- 🧾 **Document Parsing**: `pypdf2`, `python-docx` etc.
- 📚 **Vector Store**: FAISS (configurable)
- 🌐 **Frontend**: Streamlit 

## 📂 Supported File Types

- `.pdf`
- `.docx`
- `.txt`

*(More formats can be added easily)*

## 🔧 Installation

```bash
git clone https://github.com/yourusername/chat-with-documents.git
cd chat-with-documents
pip install -r requirements.txt
