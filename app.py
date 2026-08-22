import streamlit as st
import time

from src.ml_pipeline import predict_single_customer

# ==========================================
# 1. PAGE SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Marketing Intelligence Platform",
    page_icon="🎯",
    layout="wide"
)

# Initialize Session State for Chat Memory & Context
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_segment" not in st.session_state:
    st.session_state.current_segment = None
if "top_category" not in st.session_state:
    st.session_state.top_category = None

# ==========================================
# 2. MOCK BACKEND FUNCTIONS (Stub for Person 1 & 2)
# ==========================================
# Once Person 1 & 2 deliver their files, replace these with:
# from src.ml_pipeline import predict_single_customer
# from src.llm_agent import generate_campaign, chat_refinement


def mock_generate_campaign(segment_name, category, action_type):
    time.sleep(0.6)
    if action_type == "Email":
        return f"**Subject: Special Offer on {category.replace('_', ' ').title()} Just for You!**\n\nHi there,\n\nAs one of our valued **{segment_name}**, we curated our top-selling {category.replace('_', ' ')} collection for your next order. Use code **WELCOMEBACK** for 15% off at checkout!\n\nBest,\nThe Marketing Team"
    elif action_type == "Ad":
        return f"**Targeted Ad Copy ({segment_name})**\n\n*Headline:* Upgrade your {category.replace('_', ' ')} experience today.\n*Body:* Premium quality you can trust. Exclusive perks inside.\n*CTA:* Shop Now & Save"
    else:
        return f"**Strategy Brief for {segment_name} ({category}):**\n1. Trigger an automated follow-up sequence within 48 hours.\n2. Cross-sell complementary accessories in the {category} category.\n3. Offer loyalty tier incentives for their next review."

# Load external CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==========================================
# 3. HEADER (DYNAMIC LAYOUT WITH SVG)
# ==========================================
st.markdown("""
    <div class="header-container">
        <div class="header-left">
            <div class="header-icon">
                <!-- Professional Inline SVG Icon -->
                <svg xmlns="http://www.w3.org/2000/svg" width="38" height="38" viewBox="0 0 24 24" fill="#2563EB">
                    <path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z"/>
                </svg>
            </div>
            <div class="header-title-group">
                <h1>Customer Segmentation</h1>
                <h3>& Personalized Marketing Intelligence Platform</h3>
            </div>
        </div>
    </div>
    <div class="header-desc">
        Predict customer segments based on behavioral data to unlock personalized insights and determine the exact marketing approach for each specific group.
    </div>
    <hr style="margin-top: 0px; margin-bottom: 30px; border: 0; border-top: 1px solid #E2E8F0;">
""", unsafe_allow_html=True)

# ==========================================
# 4. TWO-COLUMN LAYOUT
# ==========================================
col_left, col_right = st.columns([1, 1.4], gap="large")

# --- LEFT COLUMN: INPUT PARAMETERS ---
with col_left:
    st.subheader("1. Customer Profile Input")
    st.write("Provide customer engagement metrics to classify their segment:")
    
    with st.form("customer_input_form"):
        # Changed from selectbox to text_input for universal SaaS adaptation
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
            low_review_flag = st.selectbox("Low Review Flag (<3 Stars)", options=[0, 1], index=0)
            
        predict_btn = st.form_submit_button("🔍 Predict Customer Segment", use_container_width=True)

    if predict_btn:
        with st.spinner("Classifying customer via ML..."):
            # 1. Artificial UX Delay (so the user sees the spinner)
            time.sleep(0.7) 
            
            try:
                # 2. Call the REAL machine learning pipeline
                pred = predict_single_customer(
                    recency, frequency, monetary, review_score, review_count, low_review_flag
                )
                
                # 3. Map the UI colors based on the text result
                if "Champions" in pred["segment_name"]:
                    badge_color = "green"
                elif "Risk" in pred["segment_name"]:
                    badge_color = "red"
                elif "Churned" in pred["segment_name"]:
                    badge_color = "gray"
                else:
                    badge_color = "blue"

                # 4. Save everything to session state
                st.session_state.current_segment = pred["segment_name"]
                st.session_state.top_category = top_cat if top_cat.strip() != "" else "General Merchandise"
                st.session_state.segment_desc = pred["description"]
                st.session_state.badge_color = badge_color
                st.session_state.chat_history = []
                
                
                
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
        
        # Action Buttons
        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            if st.button("📧 Email Template", use_container_width=True):
                with st.spinner("Generating email copy..."):
                    res = mock_generate_campaign(st.session_state.current_segment, st.session_state.top_category, "Email")
                    st.session_state.chat_history.append({"role": "assistant", "content": res})
        with btn_c2:
            if st.button("📢 Ad Copy", use_container_width=True):
                with st.spinner("Drafting ad campaign..."):
                    res = mock_generate_campaign(st.session_state.current_segment, st.session_state.top_category, "Ad")
                    st.session_state.chat_history.append({"role": "assistant", "content": res})
        with btn_c3:
            if st.button("🎯 Strategy Brief", use_container_width=True):
                with st.spinner("Building strategy..."):
                    res = mock_generate_campaign(st.session_state.current_segment, st.session_state.top_category, "Strategy")
                    st.session_state.chat_history.append({"role": "assistant", "content": res})
                    
        st.markdown("---")
        
        # Chat & Output History Display
        st.write("**Campaign Workspace & Conversational Refinement:**")
        chat_box = st.container(height=320)
        with chat_box:
            if len(st.session_state.chat_history) == 0:
                st.caption("Click a campaign button above or type below to start drafting.")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
        # Chat Input Box for Revisions
        if prompt := st.chat_input("Refine this campaign (e.g., 'Make it more urgent', 'Add a 20% discount')..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            # Stub reply until Person 1 plugs in real LLM router
            mock_reply = f"**Refined Copy for {st.session_state.current_segment}:**\n\nUpdated based on: *'{prompt}'*.\n\nHere is the modified version targeting {st.session_state.top_category} buyers."
            st.session_state.chat_history.append({"role": "assistant", "content": mock_reply})
            st.rerun()