"""
app.py — LifeLine AI
Streamlit Dashboard: Autonomous Emergency Blood Coordinator

Run with:
    streamlit run app.py
"""

import os
import time
from datetime import date, datetime

import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

# ── Internal modules ──────────────────────────────────────────
from database import create_database, get_donors_by_blood_group_and_city, get_all_donors
from agent import generate_broadcast_message, parse_donor_reply
from twilio_handler import send_sms


# ─────────────────────────────────────────────────────────────
# Page Config & Custom CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LifeLine AI — Emergency Blood Coordinator",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global Reset ─────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0a0d14 !important;
    color: #e2e8f0 !important;
}

/* ── Hide Streamlit branding ──────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Main content area ────────────────────────────────────── */
.main .block-container {
    padding: 1.5rem 2rem 2rem;
    max-width: 1300px;
}

/* ── Sidebar ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1320 0%, #111827 100%) !important;
    border-right: 1px solid #1e2a3a !important;
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #f87171 !important;
}

/* ── Hero banner ──────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 40%, #1e1b4b 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid #b91c1c33;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, #ef444430 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner h1 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #fff;
    margin: 0 0 .3rem;
    text-shadow: 0 2px 12px #ef444455;
}
.hero-banner p {
    color: #fca5a5;
    font-size: 1rem;
    margin: 0;
}
.pulse-dot {
    display: inline-block;
    width: 10px; height: 10px;
    background: #ef4444;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%,100% { transform: scale(1); opacity: 1; }
    50%      { transform: scale(1.6); opacity: .6; }
}

/* ── Metric cards ─────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #111827, #1a2233);
    border: 1px solid #1e2a3a;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform .2s, border-color .2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: #ef4444;
}
.metric-card .metric-value {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
}
.metric-card .metric-label {
    font-size: .8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-top: .4rem;
}
.red    { color: #ef4444; }
.green  { color: #22c55e; }
.yellow { color: #f59e0b; }
.blue   { color: #60a5fa; }

/* ── Section headers ──────────────────────────────────────── */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    padding: .5rem 0 .5rem 1rem;
    border-left: 4px solid #ef4444;
    margin: 1.5rem 0 1rem;
}

/* ── Status badge ─────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .04em;
}
.badge-contacted  { background:#1e3a5f; color:#60a5fa; }
.badge-confirmed  { background:#14532d; color:#4ade80; }
.badge-declined   { background:#450a0a; color:#f87171; }
.badge-ineligible { background:#44403c; color:#fbbf24; }
.badge-pending    { background:#1e1b4b; color:#a5b4fc; }

/* ── Chat bubbles ─────────────────────────────────────────── */
.chat-wrap { margin-bottom: 1rem; }
.bubble-donor {
    background: #1e2a3a;
    border: 1px solid #2d3f55;
    border-radius: 16px 16px 16px 4px;
    padding: .75rem 1rem;
    max-width: 70%;
    margin-bottom: .4rem;
    color: #e2e8f0;
    font-size: .9rem;
}
.bubble-ai {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border-radius: 16px 16px 4px 16px;
    padding: .75rem 1rem;
    max-width: 70%;
    margin-left: auto;
    margin-bottom: .4rem;
    color: #fff;
    font-size: .9rem;
}
.chat-label-donor { color: #94a3b8; font-size: .75rem; margin-bottom: .2rem; }
.chat-label-ai    { color: #fca5a5; font-size: .75rem; text-align:right; margin-bottom: .2rem; }

/* ── Donor table ──────────────────────────────────────────── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all .2s !important;
    padding: .6rem 1.4rem !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px #ef444455 !important;
}

/* ── Selectbox / Text input ───────────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: #111827 !important;
    border: 1px solid #1e2a3a !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Info / warning boxes ─────────────────────────────────── */
.stAlert {
    border-radius: 10px !important;
}

/* ── Divider ──────────────────────────────────────────────── */
hr { border-color: #1e2a3a !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Initialise DB & Session State
# ─────────────────────────────────────────────────────────────

# Ensure DB and table exist on every cold start
create_database()

def _init_session() -> None:
    """Initialise all required session_state keys if absent."""
    defaults = {
        # Emergency request context
        "patient_name":    "",
        "blood_group":     "",
        "hospital":        "",
        "city":            "",
        # Live donor tracking
        # List[dict]: each dict = donor row + "status" + "broadcast_msg"
        "contacted_donors": [],
        # Simulation state
        # Key = donor index; Value = {"history": [LangChain messages], "status": str}
        "sim_sessions":    {},
        "active_donor_idx": None,
        # Emergency triggered flag
        "emergency_active": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()


# ─────────────────────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
    <h1>🩸 LifeLine AI <span style="font-size:1rem;font-weight:400;color:#fca5a5;">v1.0</span></h1>
    <p><span class="pulse-dot"></span>Autonomous Emergency Blood Coordinator &mdash; Powered by Google Gemini</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR — Emergency Trigger Form
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚨 Emergency Trigger")
    st.markdown("Fill in the details below and dispatch the AI agent to locate matching donors.")
    st.markdown("---")

    patient_name = st.text_input(
        "👤 Patient Name",
        placeholder="e.g. Rahul Verma",
        key="input_patient_name",
    )
    blood_group = st.selectbox(
        "🩸 Blood Group Required",
        ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"],
        index=0,
        key="input_blood_group",
    )
    city = st.text_input(
        "📍 City",
        placeholder="e.g. Salem",
        key="input_city",
    )
    hospital = st.text_input(
        "🏥 Hospital Name",
        placeholder="e.g. Apollo Hospital",
        key="input_hospital",
    )

    st.markdown("")
    trigger_btn = st.button("🚀 DISPATCH AI AGENT", use_container_width=True, key="dispatch_btn")

    st.markdown("---")
    st.markdown("### 📋 All Registered Donors")
    all_donors_df = pd.DataFrame(get_all_donors())
    if not all_donors_df.empty:
        st.dataframe(
            all_donors_df[["name", "blood_group", "city"]].rename(columns={
                "name": "Name", "blood_group": "Blood", "city": "City"
            }),
            use_container_width=True,
            hide_index=True,
        )
    st.markdown("---")
    st.markdown(
        "<small style='color:#475569;'>LifeLine AI © 2025<br>"
        "Built with Streamlit + Google Gemini</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# DISPATCH LOGIC
# ─────────────────────────────────────────────────────────────

if trigger_btn:
    # Validate inputs
    if not patient_name.strip() or not city.strip() or not hospital.strip():
        st.sidebar.error("⚠️ Please fill in all fields before dispatching.")
    else:
        with st.spinner("🤖 AI Agent searching for matching donors and drafting messages…"):
            matched = get_donors_by_blood_group_and_city(blood_group, city)

        if not matched:
            st.sidebar.warning(
                f"No {blood_group} donors found in {city}. "
                "Try a different city or blood group."
            )
        else:
            # Save emergency context
            st.session_state["patient_name"]    = patient_name.strip()
            st.session_state["blood_group"]     = blood_group
            st.session_state["hospital"]        = hospital.strip()
            st.session_state["city"]            = city.strip()
            st.session_state["emergency_active"] = True
            st.session_state["contacted_donors"] = []
            st.session_state["sim_sessions"]     = {}
            st.session_state["active_donor_idx"] = None

            progress_bar = st.sidebar.progress(0, text="Sending messages…")
            for i, donor in enumerate(matched):
                # Generate personalised broadcast SMS via Gemini
                msg = generate_broadcast_message(
                    patient_name=patient_name.strip(),
                    blood_group=blood_group,
                    hospital=hospital.strip(),
                    city=city.strip(),
                    donor_name=donor["name"],
                )
                # Attempt to send (Twilio or console fallback)
                result = send_sms(donor["phone"], msg)

                donor_record = {
                    **donor,
                    "status":        "CONTACTED",
                    "broadcast_msg": msg,
                    "sms_mode":      result.get("mode", "console"),
                    "sms_sid":       result.get("sid", ""),
                }
                st.session_state["contacted_donors"].append(donor_record)

                # Init simulation session for this donor
                st.session_state["sim_sessions"][i] = {
                    "history": [],
                    "status":  "CONTACTED",
                }

                # Simulated delay for UX feedback
                time.sleep(0.4)
                progress_bar.progress(
                    int((i + 1) / len(matched) * 100),
                    text=f"Messaged {donor['name']} ({i+1}/{len(matched)})",
                )

            st.sidebar.success(
                f"✅ {len(matched)} donor(s) contacted! "
                "Check the dashboard below."
            )


# ─────────────────────────────────────────────────────────────
# MAIN DASHBOARD — Metrics
# ─────────────────────────────────────────────────────────────

contacted = st.session_state["contacted_donors"]

total      = len(contacted)
confirmed  = sum(1 for d in contacted if st.session_state["sim_sessions"].get(
    contacted.index(d), {}).get("status") == "CONFIRMED")
declined   = sum(1 for d in contacted if st.session_state["sim_sessions"].get(
    contacted.index(d), {}).get("status") == "DECLINED")
ineligible = sum(1 for d in contacted if st.session_state["sim_sessions"].get(
    contacted.index(d), {}).get("status") == "INELIGIBLE")
pending    = total - confirmed - declined - ineligible

col1, col2, col3, col4, col5 = st.columns(5)
metric_data = [
    (col1, total,      "Total Contacted",  "blue"),
    (col2, confirmed,  "✅ Confirmed",      "green"),
    (col3, pending,    "⏳ Pending",        "yellow"),
    (col4, declined,   "❌ Declined",       "red"),
    (col5, ineligible, "⚠️ Ineligible",    "yellow"),
]
for col, val, label, color in metric_data:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {color}">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LIVE STATUS TABLE
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">📡 Live Donor Status</div>', unsafe_allow_html=True)

if not contacted:
    st.info("🔎 No emergency has been dispatched yet. Use the sidebar to trigger an emergency alert.")
else:
    # Build a display dataframe
    rows = []
    for i, d in enumerate(contacted):
        sess_status = st.session_state["sim_sessions"].get(i, {}).get("status", "CONTACTED")
        badge_class = {
            "CONTACTED":  "badge-contacted",
            "CONFIRMED":  "badge-confirmed",
            "DECLINED":   "badge-declined",
            "INELIGIBLE": "badge-ineligible",
            "PENDING":    "badge-pending",
        }.get(sess_status, "badge-pending")

        rows.append({
            "#":              i + 1,
            "Donor Name":     d["name"],
            "Blood Group":    d["blood_group"],
            "Phone":          d["phone"],
            "City":           d["city"],
            "Last Donated":   d["last_donation_date"],
            "SMS Mode":       d.get("sms_mode", "console").upper(),
            "Status":         sess_status,
        })

    df = pd.DataFrame(rows)

    # Style the Status column with colours
    def _style_status(val):
        colours = {
            "CONTACTED":  "color:#60a5fa;font-weight:600",
            "CONFIRMED":  "color:#4ade80;font-weight:600",
            "DECLINED":   "color:#f87171;font-weight:600",
            "INELIGIBLE": "color:#fbbf24;font-weight:600",
            "PENDING":    "color:#a5b4fc;font-weight:600",
        }
        return colours.get(val, "")

    styled_df = df.style.map(_style_status, subset=["Status"])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Expand broadcast message per donor
    st.markdown('<div class="section-header">📨 Broadcast Messages Sent</div>', unsafe_allow_html=True)
    for i, d in enumerate(contacted):
        with st.expander(f"📩 Message to {d['name']} ({d['blood_group']} | {d['city']})"):
            st.markdown(
                f"<div style='background:#111827;border:1px solid #1e2a3a;"
                f"border-radius:10px;padding:1rem;color:#e2e8f0;font-style:italic;'>"
                f"{d['broadcast_msg']}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"📤 Sent via: {d.get('sms_mode','console').upper()}"
                       + (f" | SID: {d['sms_sid']}" if d.get('sms_sid') else " | (Console fallback)"))


# ─────────────────────────────────────────────────────────────
# SIMULATION SECTION — Chat with Donor Replies
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">🧪 Donor Reply Simulator</div>', unsafe_allow_html=True)

if not contacted:
    st.warning("Dispatch an emergency first to enable the reply simulator.")
else:
    # Donor selector
    donor_options = [f"{i+1}. {d['name']} ({d['blood_group']})" for i, d in enumerate(contacted)]
    selected_label = st.selectbox(
        "Select a donor to simulate their reply:",
        donor_options,
        key="donor_selector",
    )
    selected_idx = int(selected_label.split(".")[0]) - 1
    st.session_state["active_donor_idx"] = selected_idx

    active_donor = contacted[selected_idx]
    sim          = st.session_state["sim_sessions"][selected_idx]

    st.markdown(f"""
    <div style='background:#111827;border:1px solid #1e2a3a;border-radius:12px;
                padding:1rem 1.4rem;margin-bottom:1rem;'>
        <b style='color:#fca5a5;'>Simulating:</b>
        <span style='color:#e2e8f0;'>{active_donor['name']}</span>
        &nbsp;|&nbsp;
        <b style='color:#fca5a5;'>Blood Group:</b>
        <span style='color:#e2e8f0;'>{active_donor['blood_group']}</span>
        &nbsp;|&nbsp;
        <b style='color:#fca5a5;'>Status:</b>
        <span style='color:#a5b4fc;font-weight:600;'>{sim['status']}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Render chat history ───────────────────────────────────
    st.markdown("**💬 Conversation Thread:**")
    chat_container = st.container()

    with chat_container:
        # Show the broadcast message as the first AI message
        st.markdown(f"""
        <div class="chat-wrap">
            <div class="chat-label-ai">🤖 LifeLine AI</div>
            <div class="bubble-ai">{active_donor['broadcast_msg']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Render subsequent history
        for msg in sim["history"]:
            if isinstance(msg, HumanMessage):
                st.markdown(f"""
                <div class="chat-wrap">
                    <div class="chat-label-donor">👤 {active_donor['name']} (Donor)</div>
                    <div class="bubble-donor">{msg.content}</div>
                </div>
                """, unsafe_allow_html=True)
            elif isinstance(msg, AIMessage):
                st.markdown(f"""
                <div class="chat-wrap">
                    <div class="chat-label-ai">🤖 LifeLine AI</div>
                    <div class="bubble-ai">{msg.content}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Chat input (disabled once resolved) ──────────────────
    terminal_statuses = {"CONFIRMED", "DECLINED", "INELIGIBLE"}
    is_resolved = sim["status"] in terminal_statuses

    if is_resolved:
        status_colours = {
            "CONFIRMED":  ("#14532d", "#4ade80"),
            "DECLINED":   ("#450a0a", "#f87171"),
            "INELIGIBLE": ("#44403c", "#fbbf24"),
        }
        bg, fg = status_colours[sim["status"]]
        icon   = {"CONFIRMED": "✅", "DECLINED": "❌", "INELIGIBLE": "⚠️"}[sim["status"]]
        st.markdown(f"""
        <div style='background:{bg};border-radius:10px;padding:.8rem 1.2rem;
                    color:{fg};font-weight:600;text-align:center;margin-top:.5rem;'>
            {icon} This donor is <b>{sim['status']}</b>. Conversation closed.
        </div>
        """, unsafe_allow_html=True)

        if sim["status"] == "CONFIRMED":
            st.balloons()
    else:
        with st.form(key=f"reply_form_{selected_idx}", clear_on_submit=True):
            donor_reply = st.text_input(
                f"✍️ Type {active_donor['name']}'s reply here…",
                placeholder="e.g. 'Yes, I can help' or 'No, I'm unavailable'",
                key=f"reply_input_{selected_idx}",
            )
            submit_reply = st.form_submit_button("📨 Send Reply to AI Agent")

        if submit_reply and donor_reply.strip():
            with st.spinner("🧠 AI Agent is processing the reply…"):
                # Append donor's message to history
                sim["history"].append(HumanMessage(content=donor_reply.strip()))

                # Call the agent
                result = parse_donor_reply(
                    donor_message=donor_reply.strip(),
                    chat_history=sim["history"][:-1],   # history before this message
                )

                ai_response = result["ai_response"]
                new_status  = result["status"]

                # Append AI reply to history
                sim["history"].append(AIMessage(content=ai_response))

                # Update status in sim session AND contacted_donors list
                sim["status"] = new_status
                st.session_state["sim_sessions"][selected_idx]["status"] = new_status

            st.rerun()

    # ── Reset button for this donor ───────────────────────────
    if st.button("🔄 Reset this donor's conversation", key=f"reset_{selected_idx}"):
        st.session_state["sim_sessions"][selected_idx] = {
            "history": [],
            "status":  "CONTACTED",
        }
        st.rerun()


# ─────────────────────────────────────────────────────────────
# CONFIRMED DONORS PANEL
# ─────────────────────────────────────────────────────────────

confirmed_donors = [
    contacted[i]
    for i in range(len(contacted))
    if st.session_state["sim_sessions"].get(i, {}).get("status") == "CONFIRMED"
]

if confirmed_donors:
    st.markdown('<div class="section-header">🏆 Confirmed Donors — Ready to Help</div>', unsafe_allow_html=True)
    for d in confirmed_donors:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#052e16,#14532d);
                    border:1px solid #166534;border-radius:14px;
                    padding:1rem 1.5rem;margin-bottom:.8rem;'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div>
                    <span style='font-size:1.2rem;font-weight:700;color:#4ade80;'>✅ {d['name']}</span>
                    &nbsp;&nbsp;
                    <span style='background:#166534;color:#a7f3d0;padding:2px 10px;
                                 border-radius:20px;font-size:.8rem;'>{d['blood_group']}</span>
                </div>
                <div style='color:#86efac;font-size:.9rem;'>📞 {d['phone']}</div>
            </div>
            <div style='color:#6ee7b7;font-size:.85rem;margin-top:.3rem;'>
                📍 {d['city']} &nbsp;|&nbsp; Last donated: {d['last_donation_date']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.success(
        f"🎉 {len(confirmed_donors)} confirmed donor(s) ready! "
        f"Hospital: {st.session_state['hospital']} | "
        f"Patient: {st.session_state['patient_name']}"
    )


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#374151;font-size:.8rem;padding:1rem;'>
    LifeLine AI &mdash; Autonomous Emergency Blood Coordinator &nbsp;|&nbsp;
    Powered by Google Gemini &amp; LangChain &nbsp;|&nbsp;
    Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
