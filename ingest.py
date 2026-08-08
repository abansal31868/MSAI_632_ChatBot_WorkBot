import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------
# 🔑 Configuration
# ---------------------------------------------------------
FOLDER_ID = "1RbxAO9hTrz_Nac6SCE5HqQVYncvil8dD"  # Replace with your Google Drive Folder ID
HUGGINGFACE_TOKEN = "Replace with your token"  # Replace with your HF token

os.environ["HF_TOKEN"] = HUGGINGFACE_TOKEN
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_TOKEN


# ---------------------------------------------------------
# 📂 Download Files Directly via Google Drive API
# ---------------------------------------------------------
def download_drive_files(temp_dir="temp_docs"):
    os.makedirs(temp_dir, exist_ok=True)

    creds = service_account.Credentials.from_service_account_file(
        "credentials.json",
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    service = build("drive", "v3", credentials=creds)

    query = f"'{FOLDER_ID}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    files = results.get("files", [])

    downloaded_paths = []

    for f in files:
        file_id = f["id"]
        file_name = f["name"]
        print(f"📥 Downloading '{file_name}' from Google Drive...")

        request = service.files().get_media(fileId=file_id)
        file_path = os.path.join(temp_dir, file_name)

        with open(file_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()

        downloaded_paths.append(file_path)

    return downloaded_paths


# ---------------------------------------------------------
# 🚀 Parse Documents & Sync to Vector DB
# ---------------------------------------------------------
def sync_drive_to_vector_db():
    print("🔑 Authenticating and fetching files from Google Drive...")
    file_paths = download_drive_files()

    if not file_paths:
        print("❌ No files downloaded.")
        return

    documents = []
    for path in file_paths:
        print(f"📄 Parsing: {path}")
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        elif path.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            documents.extend(loader.load())

    print(f"✂️ Loaded {len(documents)} document page(s). Chunking text...")

    # Text Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Embedding & FAISS Indexing
    print("🧠 Generating embeddings and saving to FAISS...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"token": HUGGINGFACE_TOKEN}
    )

    vector_db = FAISS.from_documents(chunks, embedding_model)
    vector_db.save_local("faiss_workplace_index")

    print("✅ Vector DB created and updated successfully in 'faiss_workplace_index'!")


if __name__ == "__main__":
    sync_drive_to_vector_db()
