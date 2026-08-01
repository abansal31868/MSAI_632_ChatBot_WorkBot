import os
from langchain_community.vectorstores import FAISS
from langchain_google_community import GoogleDriveLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Environment Setup
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_hf_token_here"

FOLDER_ID = "YOUR_GOOGLE_DRIVE_FOLDER_ID"  # Folder ID from Google Drive URL


def sync_drive_to_vector_db():
    print("Loading documents from Google Drive...")
    loader = GoogleDriveLoader(
        folder_id=FOLDER_ID,
        service_account_key="credentials.json",
        recursive=True,
        file_types=[
            "application/vnd.google-apps.document",
            "application/pdf",
        ],  # Filter supported formats
    )
    documents = loader.load()

    print(f"Fetched {len(documents)} document(s). Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    print("Generating embeddings and saving to FAISS...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = FAISS.from_documents(chunks, embedding_model)
    vector_db.save_local("faiss_workplace_index")
    print("Vector DB updated successfully!")


if __name__ == "__main__":
    sync_drive_to_vector_db()