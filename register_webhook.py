import uuid
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

FOLDER_ID = "YOUR_GOOGLE_DRIVE_FOLDER_ID"
WEBHOOK_URL = (
    "https://your-public-domain.com/gdrive-webhook"  # Must be an HTTPS URL
)

creds = Credentials.from_service_account_file(
    "credentials.json",
    scopes=["https://www.googleapis.com/auth/drive.readonly"],
)
drive_service = build("drive", "v3", credentials=creds)

channel_id = str(uuid.uuid4())
body = {"id": channel_id, "type": "web_hook", "address": WEBHOOK_URL}

response = (
    drive_service.files().watch(fileId=FOLDER_ID, body=body).execute()
)
print(f"Webhook registered successfully! Channel ID: {response['id']}")