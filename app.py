from src.auth import require_login
import streamlit as st
import time

from src.ml_pipeline import predict_single_customer
from src.llm_agent import generate_action_response

# ==========================================
# 1. PAGE SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Marketing Intelligence Platform",
    page_icon="🎯",
    layout="wide"
)
#require_login()

# Initialize Session State for Chat Memory & Context
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_segment" not in st.session_state:
    st.session_state.current_segment = None
if "top_category" not in st.session_state:
    st.session_state.top_category = None

# Load external CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ==========================================
# 2. HEADER
# ==========================================

from pathlib import Path
import base64
   # st.image("AIAVENGERS.png.jpeg", width=70)
image_path = Path(__file__).parent / "AIAVENGERS.png.jpeg"

image_data = base64.b64encode(image_path.read_bytes()).decode()

header_col1, header_col2 = st.columns([0.12, 0.88])

with header_col1:
    st.markdown(
        f"""
        <img src="data:image/jpeg;base64,{image_data}"
             class="ai-avengers-logo">
        """,
        unsafe_allow_html=True
    )

with header_col2:
    st.markdown(
        """
        <h1 class="ai-title">
            <span class="lightning">⚡</span> Smart Customer Segmentation

        </h1>
        <h3 class="ai-subtitle">
            “Segment Smarter. Market Better.” <span class="lightning">⚡</span>
        </h3>
        """,
        unsafe_allow_html=True
    )
st.markdown(
    """
    <h2 style="text-align:center; color:#0F172A; margin-bottom:10px;">
        Customer Segmentation &amp; Personalized Marketing Intelligence
    </h2>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="
        text-align:center;
        color:#475569;
        font-size:18px;
        font-weight:500;
        line-height:1.6;
        max-width:1000px;
        margin:0 auto 35px auto;
    ">
        Predict customer segments based on behavioral data to unlock
        personalized insights and determine the exact marketing approach
        for each specific group.
    </p>
    """,
    unsafe_allow_html=True
)  



# ==========================================
# 3. TWO-COLUMN LAYOUT
# ==========================================
col_left, col_right = st.columns([1, 1.4], gap="large")

# --- LEFT COLUMN: INPUT PARAMETERS ---
with col_left:
    st.subheader("1. Customer Profile Input")
    st.write("Provide customer engagement metrics to classify their segment:")
    
    with st.form("customer_input_form"):
        top_cat = st.text_input(
            "Top Product Category",
            placeholder="e.g., Wireless Earbuds, Coffee Beans",
            help="Type the main product this customer buys. The AI will use this to write targeted copy."
        )
        
        c1, c2 = st.columns(2)
        with c1:
            recency = st.number_input("Recency (Days since last order)", min_value=0, max_value=1000, value=25)
            monetary = st.number_input("Total Spend ($)", min_value=0.0, max_value=10000.0, value=350.0, step=10.0)
            review_count = st.number_input("Review Count", min_value=0, max_value=20, value=2)
        with c2:
            frequency = st.number_input("Frequency (Total Orders)", min_value=1, max_value=50, value=3)
            review_score = st.slider("Average Review Score", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
            # Removed low_review_flag from the UI to make it universally adaptable!
            
        predict_btn = st.form_submit_button("🔍 Predict Customer Segment", use_container_width=True)

    if predict_btn:
        with st.spinner("Classifying customer via ML..."):
            time.sleep(0.4) 
            
            try:
                # 1. Call the ML pipeline
                pred = predict_single_customer(
                    recency, frequency, monetary, review_score, review_count, 0
                )
                
                # 2. Map the UI colors
                if "Champions" in pred["segment_name"]:
                    badge_color = "green"
                elif "Risk" in pred["segment_name"]:
                    badge_color = "red"
                elif "Churned" in pred["segment_name"]:
                    badge_color = "gray"
                else:
                    badge_color = "blue"

                # 3. Save everything to session state
                st.session_state.current_segment = pred["segment_name"]
                st.session_state.top_category = top_cat if top_cat.strip() != "" else "General Merchandise"
                st.session_state.segment_desc = pred["description"]
                st.session_state.badge_color = badge_color
                st.session_state.chat_history = [] 
                
                # --- NEW: THE AUTO-TRIGGERED LLM RESPONSE ---
                with st.spinner("Generating initial AI strategy overview..."):
                    initial_prompt = "Provide a brief, high-level overview of exactly how we should market to this specific segment."
                    _, st.session_state.chat_history = generate_action_response(
                        st.session_state.chat_history, 
                        st.session_state.current_segment, 
                        st.session_state.top_category, 
                        initial_prompt
                    )
                
            except Exception as e:
                st.error(f"Backend Error: {e}")

    # Display Active Classification Status
    if st.session_state.current_segment:
        st.success(f"**Identified Segment:** {st.session_state.current_segment}")
        st.info(f"**Category:** `{st.session_state.top_category}` | {st.session_state.segment_desc}")

# --- RIGHT COLUMN: CAMPAIGN GENERATOR & CHAT ---
with col_right:
    st.subheader("2. Actionable Campaign Intelligence")
    
    if not st.session_state.current_segment:
        st.info("👈 Enter customer details and click **'Predict Customer Segment'** to unlock tailored campaigns.")
    else:
        st.write("Generate tailored marketing collateral:")
        
        # Action Buttons (Now wired to the LLM agent!)
        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            if st.button("📧 Email Template", use_container_width=True):
                with st.spinner("Generating email copy..."):
                    prompt = "Write a high-converting marketing email template (with a subject line) for this segment. Give them a compelling reason to buy again today."
                    _, st.session_state.chat_history = generate_action_response(
                        st.session_state.chat_history, st.session_state.current_segment, st.session_state.top_category, prompt
                    )
                    st.rerun() # Refreshes the UI instantly to show the chat
                    
        with btn_c2:
            if st.button("📢 Ad Copy", use_container_width=True):
                with st.spinner("Drafting ad campaign..."):
                    prompt = "Draft short, punchy Facebook/Instagram Ad copy for this segment. Include a Headline, Body Text, and CTA."
                    _, st.session_state.chat_history = generate_action_response(
                        st.session_state.chat_history, st.session_state.current_segment, st.session_state.top_category, prompt
                    )
                    st.rerun()
                    
        with btn_c3:
            if st.button("🎯 Strategy Brief", use_container_width=True):
                with st.spinner("Building strategy..."):
                    prompt = "Provide a detailed, bulleted 3-step retention strategy and follow-up sequence for this segment."
                    _, st.session_state.chat_history = generate_action_response(
                        st.session_state.chat_history, st.session_state.current_segment, st.session_state.top_category, prompt
                    )
                    st.rerun()
                    
        st.markdown("---")
        
        # Chat & Output History Display
        st.write("**Campaign Workspace & Conversational Refinement:**")
        chat_box = st.container(height=320)
        with chat_box:
            if len(st.session_state.chat_history) == 0:
                st.caption("Click a campaign button above or type below to start drafting.")
            
            # Loop through history, but skip the hidden "system" prompts
            for msg in st.session_state.chat_history:
                if msg["role"] != "system":
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                    
        # Chat Input Box for Revisions
        if prompt := st.chat_input("Refine this campaign (e.g., 'Make it more urgent', 'Add a 20% discount')..."):
            with st.spinner("Refining..."):
                _, st.session_state.chat_history = generate_action_response(
                    st.session_state.chat_history, st.session_state.current_segment, st.session_state.top_category, prompt
                )
                st.rerun()