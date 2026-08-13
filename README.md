# 💼 Workplace RAG Copilot

An enterprise workplace productivity assistant built with **LangChain**, **Hugging Face**, **FAISS**, and **Streamlit**. It integrates directly with **Google Drive** using **FastAPI webhooks** for real-time document synchronization and retrieval-augmented generation (RAG).

---

## 🏗️ Architecture Overview

1. **Ingestion (`ingest.py`)**: Fetches documents from a Google Drive folder, splits them into semantic chunks, generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`, and builds a local FAISS vector store.
2. **Real-time Synchronization (`webhook_server.py` & `register_webhook.py`)**: Subscribes to Google Drive push notifications to automatically update the vector index whenever documents are added or updated.
3. **Chat Interface (`app.py`)**: Streamlit web interface powered by Hugging Face Inference API (`meta-llama/Llama-3.1-8B-Instruct`, via `ChatHuggingFace`) for answering workplace queries.
4. **Personalization (`personalization.py`)**: A sidebar form (name, department, preferred answer style) saved locally in SQLite and folded into the system prompt on every turn -- no extra model calls.
5. **Memory Management (`memory.py`)**: Short-term memory replays the last few turns into the model's context via a `chat_history` placeholder; long-term memory persists facts across sessions when the user explicitly says "remember that ...".
6. **Multilingual Support (`i18n.py`)**: Detects the incoming message's language (`langdetect`) and translates to/from English (`deep-translator`, via Google Translate's free endpoint) around the English-only RAG/router core.
7. **Task Automation / API & Tool Integration (`router.py`, `tools.py`, `calendar_integration.py`, `email_integration.py`)**: A deterministic keyword/regex router checks each message for a to-do, calendar, or email request *before* it reaches the RAG chain, so these actions cost zero extra model calls and work even if the Hugging Face model/provider is unavailable. To-dos are logged to a local SQLite table. Calendar events are created on the user's real Google Calendar via OAuth when configured, falling back to a local `.ics` file otherwise. Email has two distinct intents: "draft an email..." always just saves an LLM-written draft locally and never sends anything, while "send an email..." attempts a real send via Gmail SMTP — but for demo safety, a real send is always routed to one fixed, pre-configured address rather than whatever recipient was parsed from the message (the parsed recipient is still shown inside the email so the extraction is visible), and it falls back to a local draft if sending isn't configured. See the "Why deterministic, not LLM tool-calling" note in `router.py` and `test_tool_calling.py` for the reliability testing behind this design choice.

---

## 📂 Project Structure

```text
├── app.py                 # Streamlit chat interface frontend
├── ingest.py              # Google Drive ingestion & vector store builder
├── webhook_server.py      # FastAPI server to listen for Drive update events
├── register_webhook.py    # Script to register webhook channel with Google Drive
├── router.py               # Deterministic intent router (task automation + remember)
├── tools.py                 # Local, free task-automation tools (to-do, calendar, email draft)
├── memory.py                # Short-term + long-term conversation memory (Feature #6)
├── personalization.py       # Local user-profile personalization (Feature #5)
├── i18n.py                  # Language detection + translation (Feature #15)
├── credentials.json        # Google Cloud Service Account credentials (DO NOT COMMIT)
├── faiss_workplace_index/  # Generated local FAISS vector index database
├── workbot_data.db         # Local SQLite store (to-dos, long-term facts, profile)
└── requirements.txt         # Project dependencies
├── app.py                    # Streamlit chat interface frontend
├── ingest.py                 # Google Drive ingestion & vector store builder
├── webhook_server.py         # FastAPI server to listen for Drive update events
├── register_webhook.py       # Script to register webhook channel with Google Drive
├── router.py                 # Deterministic intent router for Task Automation
├── tools.py                  # To-do / calendar (local) / email-draft tool functions
├── calendar_integration.py   # Real Google Calendar (OAuth) integration, with local fallback
├── email_integration.py      # Real email sending (Gmail SMTP), always routed to a fixed demo address
├── test_tool_calling.py      # Standalone LLM tool-calling reliability check (not part of the app)
├── credentials.json          # Google Cloud Service Account credentials (DO NOT COMMIT)
├── oauth_credentials.json    # Google OAuth Desktop client for Calendar (DO NOT COMMIT)
├── calendar_token.json       # Cached OAuth token, created after first Calendar auth (DO NOT COMMIT)
├── faiss_workplace_index/    # Generated local FAISS vector index database
├── workbot_data.db           # Local SQLite store for to-dos
├── workbot_calendar.ics      # Local calendar fallback file
├── email_drafts/             # Saved (unsent) email drafts
└── requirements.txt          # Project dependencies

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
pip install langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers transformers accelerate torch streamlit google-api-python-client google-auth-httplib2 google-auth-oauthlib "langchain-google-community[drive]" fastapi uvicorn pypdf langchain-classic dateparser langdetect deep-translator
🔑 Configuration
Step 1: Google Cloud Service Account
Go to Google Cloud Console > IAM & Admin > Service Accounts.

Create a Service Account (e.g., rag-drive-reader) and download the key file as JSON.

Save the key file as credentials.json in the root directory.

Locate the client_email inside credentials.json and share your target Google Drive folder with that email (give Viewer permission).

Step 2: Environment Variables
Export your Hugging Face API Token in your terminal or set it in your scripts:

Bash
export HUGGINGFACEHUB_API_TOKEN="your_hf_token_here"

Step 3: Enable Hugging Face Inference Providers
`app.py` calls the chat model through Hugging Face's Inference Providers routing, which requires at least one provider to be enabled on your account before any request will succeed:
- Go to [huggingface.co/settings/inference-providers](https://huggingface.co/settings/inference-providers).
- Enable at least one provider that serves `meta-llama/Llama-3.1-8B-Instruct` — Featherless AI, Novita, and Nscale currently do, are free/pay-as-you-go through Hugging Face, and don't require a separate account with that provider.
- Without this step, `streamlit run app.py` will fail with `is not supported by any provider you have enabled` even though your token and code are correct.

Step 4: Ensure the Google Drive API is Enabled
- Open the Google Cloud Console API Library.
- Ensure your project (project-msai-rag-drive-reader or equivalent) is selected at the top.
- Search for Google Drive API and click Enable.

Step 5: Set Up Google Calendar Integration (optional, for real calendar events)
Calendar events default to a local `.ics` file. To have WorkBot create real events on your own Google Calendar instead:
- In the same Cloud project, open the API Library, search for Google Calendar API, and click Enable.
- Go to APIs & Services → Credentials → Create Credentials → OAuth client ID. Choose application type "Desktop app." Download the resulting JSON and save it in the repo root as `oauth_credentials.json` (this is separate from `credentials.json`, which is the service account used for Drive).
- Go to APIs & Services → OAuth consent screen and add your own Google account under "Test users" (the app is unverified, so only test users can complete the consent flow).
- The first time a calendar request runs, a browser window will open asking you to log in and approve calendar access. After that, a `calendar_token.json` is cached locally and you won't be prompted again.
- If any of this isn't set up, or the request fails for any reason, WorkBot automatically falls back to writing the event to the local `workbot_calendar.ics` file instead — this integration is additive, not required.

Step 6: Set Up Real Email Sending (optional, for real "send an email" requests)
"Draft an email..." always just saves a local `.txt` file and never sends anything — no setup needed for that. "Send an email..." is a separate, real send via Gmail SMTP, and needs its own setup:
- On the sending Gmail account, turn on 2-Step Verification (required for App Passwords): [myaccount.google.com/security](https://myaccount.google.com/security).
- Generate an App Password at [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) — pick any name (e.g. "WorkBot"), and copy the 16-character password it gives you.
- Set these as environment variables before running the app:
```bash
export GMAIL_SENDER_ADDRESS="your_gmail_address@gmail.com"
export GMAIL_APP_PASSWORD="the_16_char_app_password"
# Optional: send demo emails somewhere other than your own inbox
export WORKBOT_DEMO_RECIPIENT="your_gmail_address@gmail.com"
```
- **Demo safety, by design:** whatever recipient the router parses out of a message (e.g. "send an email to John about...") is never used as the real `To:` address. Every real send goes to `WORKBOT_DEMO_RECIPIENT` (or `GMAIL_SENDER_ADDRESS` if that's not set) instead — the parsed recipient is preserved inside the email body so the extraction is still visible. This means a live demo can't accidentally mail a real third party, no matter what gets typed or misparsed.
- No Google Cloud Console project or OAuth client is needed for this — it's a separate, simpler auth path from the Calendar integration above.
- If the environment variables aren't set, or the send fails for any reason, WorkBot automatically falls back to saving a local draft instead — this integration is additive, not required.

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

## 🧩 Using Personalization, Memory, and Multilingual Support

- **Personalization**: open the sidebar, fill in your name/department/preferred answer style, and click Save. It applies starting with your next message.
- **Long-term memory**: say something like *"remember that I work on the Q3 migration project"*. WorkBot will recall it in the system prompt on every later turn, in this and future sessions, until you clear `workbot_data.db`.
- **Short-term memory**: no action needed -- the last few turns are automatically replayed into context so follow-up questions ("what about the 2024 version?") work without repeating yourself.
- **Multilingual support**: just type in another language. WorkBot detects it, answers using the same English-only document index and router, and translates the reply back automatically. If Google Translate's free endpoint is unreachable or rate-limited, WorkBot falls back to answering in English rather than failing.

🩹 Troubleshooting

**Translation looks wrong or isn't happening**
`i18n.py` uses `deep-translator`'s `GoogleTranslator`, which calls Google Translate's free web endpoint (no API key). It can rate-limit under heavy use; on any error it fails open and returns the original text, so a translation hiccup degrades to an English-only reply rather than crashing the chat. If this happens often in a demo, test the specific language pair with a short standalone script before relying on it live.

**A short message got treated as English when it wasn't**
`langdetect` is unreliable on very short strings, so anything under 12 characters is assumed English by design (see `i18n.MIN_CHARS_FOR_DETECTION`) rather than risking a wrong-language guess on something like "ok" or "sí". Longer messages detect correctly.


**`ValueError: Model ... is not supported for task text-generation and provider ...`**
Some models on Hugging Face's Inference Providers routing are only served via the *conversational* (chat) endpoint, not raw text-generation. `app.py` already handles this by wrapping the LLM in `ChatHuggingFace` rather than calling `HuggingFaceEndpoint` directly — if you're extending `app.py` and hit this error on a new model, wrap it the same way:
```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(repo_id="...", task="text-generation", provider="auto")
llm = ChatHuggingFace(llm=llm)
```

**`BadRequestError: ... is not supported by any provider you have enabled`**
This means the provider step in Configuration (Step 3) hasn't been done for your account, or the model you're using isn't served by any provider you've enabled. Before assuming the code is broken, check a model's live provider mapping:
```bash
curl "https://huggingface.co/api/models/<repo_id>?expand[]=inferenceProviderMapping"
```
`meta-llama/Llama-3.2-3B-Instruct`, for example, is only mapped to one provider (Featherless AI), which is why the app defaults to `meta-llama/Llama-3.1-8B-Instruct` instead — it's live across four providers, so it's far less likely to hit this error.

**"Send an email..." always falls back to a local draft instead of actually sending**
This means `email_integration.try_send_email` returned `None`, which happens if `GMAIL_SENDER_ADDRESS`/`GMAIL_APP_PASSWORD` aren't set, or the SMTP login/send itself failed (e.g. a regular Gmail password instead of an App Password, 2-Step Verification not enabled, or a typo in the app password). Nothing crashes — it's designed to degrade to draft-only — but if you expected a real send, double check Configuration Step 6 and that you're using the 16-character App Password, not your normal Gmail login password.

🔒 Security Best Practices
.gitignore Note: Ensure credentials.json, oauth_credentials.json, calendar_token.json, and .env are added to your .gitignore file to prevent leaking secrets to GitHub. workbot_data.db, workbot_calendar.ics, and email_drafts/ contain locally-generated demo data and should probably be gitignored too, rather than committed. GMAIL_APP_PASSWORD is an environment variable, not a file, so there's nothing to gitignore for it — but treat it like any other secret: don't paste it into code, commit messages, or shared scripts, and revoke it from [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) if it's ever exposed.

Email Send Safety: `email_integration.py` always routes real sends to a fixed `WORKBOT_DEMO_RECIPIENT`/`GMAIL_SENDER_ADDRESS` address regardless of the parsed recipient (see Configuration Step 6). Don't remove that override to make "true" arbitrary-recipient sending work without adding a confirmation step first — the whole point is that a live demo can't accidentally email a real third party.

Access Control: For sensitive workplace folders (e.g., HR or Payroll), maintain separate vector database indexes or apply role-based permission checks.

📜 License
Distributed under the MIT License.
