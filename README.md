# 💼 Workplace RAG Copilot

An enterprise workplace productivity assistant built with **LangChain**, **Hugging Face**, **FAISS**, and **Streamlit**. It integrates directly with **Google Drive** using **FastAPI webhooks** for real-time document synchronization and retrieval-augmented generation (RAG).

---

## 🏗️ Architecture Overview

1. **Ingestion (`ingest.py`)**: Fetches documents from a Google Drive folder, splits them into semantic chunks, generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`, and builds a local FAISS vector store.
2. **Real-time Synchronization (`webhook_server.py` & `register_webhook.py`)**: Subscribes to Google Drive push notifications to automatically update the vector index whenever documents are added or updated.
3. **Chat Interface (`app.py`)**: Streamlit web interface powered by Hugging Face Inference API (`meta-llama/Llama-3.2-3B-Instruct`) for answering workplace queries.

---

## 📂 Project Structure

```text
├── app.py                 # Streamlit chat interface frontend
├── ingest.py              # Google Drive ingestion & vector store builder
├── webhook_server.py      # FastAPI server to listen for Drive update events
├── register_webhook.py    # Script to register webhook channel with Google Drive
├── credentials.json       # Google Cloud Service Account credentials (DO NOT COMMIT)
├── faiss_workplace_index/ # Generated local FAISS vector index database
└── requirements.txt       # Project dependencies

🛠️ Prerequisites & Setup
1. Requirements
macOS / Linux / Windows with Python 3.10 – 3.12 installed.

Google Cloud Console account with Google Drive API enabled.

Hugging Face Account Token (HUGGINGFACEHUB_API_TOKEN).

2. Installation
Clone the repository and install dependencies inside your virtual environment:

Bash
git clone [https://github.com/your-username/workplace-rag-copilot.git](https://github.com/your-username/workplace-rag-copilot.git)
cd workplace-rag-copilot

# Create and activate virtual environment (optional)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install --upgrade pip
pip install langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers transformers accelerate torch streamlit google-api-python-client google-auth-httplib2 google-auth-oauthlib "langchain-google-community[drive]" fastapi uvicorn pypdf 
🔑 Configuration
Step 1: Google Cloud Service Account
Go to Google Cloud Console > IAM & Admin > Service Accounts.

Create a Service Account (e.g., rag-drive-reader) and download the key file as JSON.

Save the key file as credentials.json in the root directory.

Locate the client_email inside credentials.json and share your target Google Drive folder with that email (give Viewer permission).

Step 2: Environment Variables
Export your Hugging Face API Token in your terminal or set it in your scripts:

Step 3: Ensure the Google Drive API is Enabled
- Open the Google Cloud Console API Library.
- Ensure your project (project-msai-rag-drive-reader or equivalent) is selected at the top.
- Search for Google Drive API and click Enable.

Bash
export HUGGINGFACEHUB_API_TOKEN="your_hf_token_here"
🚀 How to Run
1. Run Initial Ingestion
To build the initial FAISS index from your Google Drive folder, configure FOLDER_ID in ingest.py and run:

Bash
python ingest.py
2. Launch the Streamlit Chatbot UI
Run the interactive workplace assistant web application:

Bash
streamlit run app.py
Navigate to http://localhost:8501 in your browser.

3. Setup Real-Time Webhook Updates (Optional)
To enable instant re-indexing when Google Drive files change:

Start the FastAPI listener:

Bash
uvicorn webhook_server:app --host 0.0.0.0 --port 8000
In a separate terminal, expose port 8000 via HTTPS (e.g., using ngrok http 8000).

Update WEBHOOK_URL in register_webhook.py with your public HTTPS endpoint and run:

Bash
python register_webhook.py

🔒 Security Best Practices
.gitignore Note: Ensure credentials.json and .env are added to your .gitignore file to prevent leaking secrets to GitHub.

Access Control: For sensitive workplace folders (e.g., HR or Payroll), maintain separate vector database indexes or apply role-based permission checks.

📜 License
Distributed under the MIT License.
