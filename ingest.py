import os
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

FOLDER_ID = "1RbxAO9hTrz_Nac6SCE5HqQVYncvil8dD"  # Folder ID from Google Drive URL

def download_and_load_pdfs(folder_id, credentials_path):
    print("Connecting to Google Drive API...")
    creds = service_account.Credentials.from_service_account_file(credentials_path)
    service = build('drive', 'v3', credentials=creds)

    # List files inside the folder
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name, mimeType)"
    ).execute()
    
    files = results.get('files', [])
    temp_dir = "./temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    
    all_docs = []
    
    for file in files:
        file_id = file['id']
        file_name = file['name']
        print(f"Downloading: {file_name}")
        
        # Download file content
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(os.path.join(temp_dir, file_name), 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
            
        # Load local PDF using PyPDFLoader
        loader = PyPDFLoader(os.path.join(temp_dir, file_name))
        all_docs.extend(loader.load())
        
    return all_docs

def sync_drive_to_vector_db():
    print("Loading documents from Google Drive...")
    documents = download_and_load_pdfs(FOLDER_ID, "credentials.json")

    print(f"Fetched {len(documents)} page(s). Chunking text...")
    if not documents:
        print("⚠️ No documents were found in the specified Google Drive folder.")
        return
    
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