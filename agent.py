"""
agent.py — LifeLine AI
Core LangChain + OpenRouter agent logic.

OpenRouter (https://openrouter.ai) is an OpenAI-compatible gateway that
routes requests to 200+ models (Gemini, Claude, Llama, Mistral, etc.).
We use langchain-openai's ChatOpenAI with a custom base_url.

Responsibilities:
  1. generate_broadcast_message()  — craft the initial SMS blast to donors.
  2. parse_donor_reply()           — autonomously handle donor replies
                                     (confirmed / rejected / needs-more-info).
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── Load environment variables ────────────────────────────────
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ── OpenRouter configuration ──────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models to try in order — first free, then paid fallbacks.
# Free models on OpenRouter are marked with ":free" suffix.
_MODEL_CANDIDATES = [
    "google/gemini-2.0-flash-exp:free",       # Free Gemini 2.0 Flash
    "google/gemini-flash-1.5",                # Gemini 1.5 Flash (paid tier)
    "meta-llama/llama-3.3-70b-instruct:free", # Free Llama 3.3 70B
    "mistralai/mistral-7b-instruct:free",     # Free Mistral 7B fallback
    "openai/gpt-4o-mini",                     # Paid GPT-4o-mini final fallback
]

# ── Extra headers OpenRouter recommends ──────────────────────
_OR_HEADERS = {
    "HTTP-Referer": "https://lifeline-ai.app",  # Your site / app URL
    "X-Title":      "LifeLine AI",              # Shown in openrouter.ai dashboard
}


# ─────────────────────────────────────────────────────────────
# LLM Initialisation
# ─────────────────────────────────────────────────────────────

def _get_llm(model: str = "google/gemini-2.0-flash-exp:free") -> ChatOpenAI:
    """
    Returns a ChatOpenAI instance pointed at OpenRouter's API.
    Raises ValueError if the API key is absent.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. "
            "Please add it to your .env file and restart the app.\n"
            "Get a free key at: https://openrouter.ai/keys"
        )
    return ChatOpenAI(
        model=model,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.3,
        default_headers=_OR_HEADERS,
    )


def _get_llm_with_fallback() -> ChatOpenAI:
    """
    Tries each model in _MODEL_CANDIDATES sequentially.
    Returns the first one that can be initialised.
    Raises RuntimeError only if all candidates fail.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. "
            "Add it to your .env file and restart."
        )
    # All ChatOpenAI instances are lazy — they don't call the API at init time.
    # Return the primary model; actual errors will surface on .invoke().
    # The try/except in the public functions catches invoke-time errors.
    return _get_llm(_MODEL_CANDIDATES[0])


# ─────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────

_BROADCAST_SYSTEM = (
    "You are LifeLine AI, an autonomous emergency blood coordinator. "
    "Your job is to draft a concise, urgent, yet warm and respectful SMS to a registered "
    "blood donor to request their help. Keep it under 160 characters so it fits a single SMS. "
    "Include the patient's details and a clear call to action (reply YES/NO). "
    "Return ONLY the SMS text — no extra commentary or quotation marks."
)

_BROADCAST_HUMAN = (
    "Generate an emergency broadcast SMS for the following request:\n"
    "- Patient Name  : {patient_name}\n"
    "- Blood Group   : {blood_group}\n"
    "- Hospital      : {hospital}\n"
    "- City          : {city}\n"
    "- Donor Name    : {donor_name}\n\n"
    "Draft the SMS now. Be urgent but polite."
)

broadcast_prompt = ChatPromptTemplate.from_messages([
    ("system", _BROADCAST_SYSTEM),
    ("human",  _BROADCAST_HUMAN),
])

# --- Reply parser / follow-up prompt ---
_REPLY_SYSTEM = (
    "You are LifeLine AI, an autonomous emergency blood coordinator. "
    "You are having a text conversation with a potential blood donor who received an "
    "emergency broadcast SMS.\n\n"
    "Your goals:\n"
    "1. If the donor says YES / agrees to donate:\n"
    "   - Thank them warmly.\n"
    '   - Ask: "Have you donated blood in the last 3 months? (Yes/No)"\n'
    "   - If they say NO to that → they are eligible, mark them CONFIRMED.\n"
    "   - If they say YES to that → politely explain they must wait 90 days, "
    "mark them INELIGIBLE.\n"
    "2. If the donor says NO / declines → thank them gracefully, mark them DECLINED.\n"
    "3. If the reply is ambiguous → ask a polite clarifying question.\n\n"
    "ALWAYS end your response with exactly one of these status tags on its own line:\n"
    "  [STATUS: CONFIRMED]\n"
    "  [STATUS: DECLINED]\n"
    "  [STATUS: INELIGIBLE]\n"
    "  [STATUS: PENDING]\n\n"
    "Use [STATUS: PENDING] when you still need more information.\n"
    "Keep your replies short (2-3 sentences), warm, and professional."
)

_REPLY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _REPLY_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human",  "{donor_message}"),
])


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def generate_broadcast_message(
    patient_name: str,
    blood_group: str,
    hospital: str,
    city: str,
    donor_name: str,
) -> str:
    """
    Uses the LLM (via OpenRouter) to craft a personalised emergency broadcast SMS.

    Returns the AI-generated message string, or a plain-text fallback
    if every API call fails — so the app never crashes.
    """
    for model in _MODEL_CANDIDATES:
        try:
            llm = _get_llm(model)
            messages = broadcast_prompt.format_messages(
                patient_name=patient_name,
                blood_group=blood_group,
                hospital=hospital,
                city=city,
                donor_name=donor_name,
            )
            response = llm.invoke(messages)
            text = response.content.strip()
            if text:
                return text
        except Exception:
            continue  # Try next model

    # All models failed — return a clear, plain-text fallback SMS
    return (
        f"URGENT: {blood_group} blood needed for {patient_name} at "
        f"{hospital}, {city}. Dear {donor_name}, can you help? "
        f"Reply YES or NO. – LifeLine AI"
    )


def parse_donor_reply(donor_message: str, chat_history: list) -> dict:
    """
    Autonomously handles a donor's reply message using the LLM.

    Args:
        donor_message:  The raw text the donor sent back (simulated in the UI).
        chat_history:   List of LangChain HumanMessage / AIMessage objects.

    Returns:
        {
            "ai_response": str,    # What LifeLine AI says next
            "status":      str,    # CONFIRMED | DECLINED | INELIGIBLE | PENDING
        }
    """
    for model in _MODEL_CANDIDATES:
        try:
            llm = _get_llm(model)
            messages = _REPLY_PROMPT.format_messages(
                chat_history=chat_history,
                donor_message=donor_message,
            )
            response = llm.invoke(messages)
            raw_text: str = response.content.strip()

            if not raw_text:
                continue  # Empty response — try next model

            # ── Extract status tag ────────────────────────────
            status = "PENDING"
            for tag in ("CONFIRMED", "DECLINED", "INELIGIBLE", "PENDING"):
                if f"[STATUS: {tag}]" in raw_text:
                    status = tag
                    break

            # Strip the status tag from the visible reply text
            display_text = raw_text
            for tag in ("CONFIRMED", "DECLINED", "INELIGIBLE", "PENDING"):
                display_text = display_text.replace(f"[STATUS: {tag}]", "").strip()

            return {"ai_response": display_text, "status": status}

        except Exception:
            continue  # Try next model

    # All models exhausted
    return {
        "ai_response": (
            "I'm having trouble connecting to the AI service right now. "
            "Please try again in a moment."
        ),
        "status": "PENDING",
    }
