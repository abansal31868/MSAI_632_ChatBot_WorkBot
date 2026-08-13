"""
Real email sending for WorkBot's Task Automation feature, via Gmail SMTP.

Demo safety: unlike calendar_integration.py, this module never sends to
whatever recipient the router parsed out of a free-text message. It always
sends to one fixed, pre-configured address (WORKBOT_DEMO_RECIPIENT, or the
sender's own address if that's not set). The parsed recipient is preserved
in the email itself (a "Originally addressed to" line) so the extraction
is still visible in the demo -- it just can't accidentally mail a real
third party if the parse is wrong or the wrong name gets typed live.

One-time setup required before this will do anything (see README):
1. Turn on 2-Step Verification on the sending Gmail account.
2. Generate an App Password at myaccount.google.com/apppasswords.
3. Set GMAIL_SENDER_ADDRESS and GMAIL_APP_PASSWORD as environment
   variables. Optionally set WORKBOT_DEMO_RECIPIENT to send demo emails
   somewhere other than the sender's own inbox.

No Google Cloud Console project, OAuth client, or new API needs to be
enabled for this -- it's a different, simpler auth path than
calendar_integration.py's OAuth flow.

If any of this isn't set up, or the send fails for any reason, this module
returns None so the caller (tools.py) can fall back to saving a local
draft instead of breaking the conversation.
"""

import os
import smtplib
from email.message import EmailMessage

SENDER_ENV_VAR = "GMAIL_SENDER_ADDRESS"
APP_PASSWORD_ENV_VAR = "GMAIL_APP_PASSWORD"
DEMO_RECIPIENT_ENV_VAR = "WORKBOT_DEMO_RECIPIENT"

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


def try_send_email(parsed_recipient: str, subject: str, body: str) -> str | None:
    """Attempt to actually send an email via Gmail SMTP.

    `parsed_recipient` is whatever the router extracted from the user's
    message -- it is NOT used as the real To: address. The real send always
    goes to a fixed demo address for safety; `parsed_recipient` is only
    included in the email body so the extraction is visible.

    Returns a confirmation string on success, or None if this couldn't be
    done for any reason -- missing setup or a send failure. Callers should
    treat None as "fall back to saving a local draft," not as an error to
    surface to the user.
    """
    sender = os.environ.get(SENDER_ENV_VAR)
    app_password = os.environ.get(APP_PASSWORD_ENV_VAR)
    if not sender or not app_password:
        return None

    demo_recipient = os.environ.get(DEMO_RECIPIENT_ENV_VAR, sender)

    msg = EmailMessage()
    msg["Subject"] = f"[WorkBot demo] {subject.strip()}"
    msg["From"] = sender
    msg["To"] = demo_recipient
    msg.set_content(
        f"(DEMO MODE: this message was actually addressed to \"{parsed_recipient.strip()}\" "
        f"by the request below, but WorkBot always routes real sends to this fixed "
        f"demo address instead, so nothing goes to an unintended recipient.)\n\n"
        f"{'-' * 60}\n\n"
        f"{body}"
    )

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(sender, app_password)
            server.send_message(msg)
    except Exception:
        return None

    return (
        f"Sent! (Demo safety: real sends always go to **{demo_recipient}** instead of "
        f"the parsed recipient, **{parsed_recipient.strip()}** -- that's noted inside "
        f"the email itself.)"
    )
