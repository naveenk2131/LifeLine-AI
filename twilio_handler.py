"""
twilio_handler.py — LifeLine AI
Wraps Twilio SMS sending with a graceful fallback.

If Twilio credentials are absent or invalid, the message is printed to
the console so the app never crashes during local testing.
"""

import os
from dotenv import load_dotenv

load_dotenv()

TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID",  "")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN",    "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER",  "")


def _twilio_configured() -> bool:
    """Returns True only if all three Twilio credentials are present."""
    return bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER)


def send_sms(to_phone: str, message: str) -> dict:
    """
    Sends an SMS via Twilio to `to_phone` with `message`.

    Args:
        to_phone: Recipient E.164 phone number (e.g. "+919876543210").
        message:  Body of the SMS (keep under 160 chars for a single segment).

    Returns:
        {
            "success": bool,
            "sid":     str | None,   # Twilio message SID on success
            "error":   str | None,   # Error description on failure
            "mode":    "twilio" | "console",
        }
    """
    if not _twilio_configured():
        # ── Console fallback (for local testing without Twilio creds) ──
        print("\n" + "=" * 60)
        print("[LIFELINE AI — CONSOLE SMS FALLBACK]")
        print(f"  TO     : {to_phone}")
        print(f"  MESSAGE: {message}")
        print("=" * 60 + "\n")
        return {
            "success": True,
            "sid":     None,
            "error":   None,
            "mode":    "console",
        }

    # ── Live Twilio send ──────────────────────────────────────
    try:
        from twilio.rest import Client

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_phone,
        )
        return {
            "success": True,
            "sid":     msg.sid,
            "error":   None,
            "mode":    "twilio",
        }
    except Exception as exc:
        # Surface the error but never crash the caller
        error_msg = str(exc)
        print(f"[Twilio Error] {error_msg}")
        return {
            "success": False,
            "sid":     None,
            "error":   error_msg,
            "mode":    "twilio",
        }


def send_whatsapp(to_phone: str, message: str) -> dict:
    """
    Sends a WhatsApp message via Twilio Sandbox.
    Requires the recipient to have joined the Twilio WhatsApp sandbox.

    Internally calls send_sms with Twilio's WhatsApp prefix format.
    """
    whatsapp_to   = f"whatsapp:{to_phone}"
    whatsapp_from = f"whatsapp:{TWILIO_PHONE_NUMBER}" if TWILIO_PHONE_NUMBER else ""

    if not _twilio_configured():
        print("\n" + "=" * 60)
        print("[LIFELINE AI — CONSOLE WHATSAPP FALLBACK]")
        print(f"  TO     : {whatsapp_to}")
        print(f"  MESSAGE: {message}")
        print("=" * 60 + "\n")
        return {"success": True, "sid": None, "error": None, "mode": "console"}

    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=whatsapp_from,
            to=whatsapp_to,
        )
        return {"success": True, "sid": msg.sid, "error": None, "mode": "twilio"}
    except Exception as exc:
        error_msg = str(exc)
        print(f"[Twilio WhatsApp Error] {error_msg}")
        return {"success": False, "sid": None, "error": error_msg, "mode": "twilio"}
