import streamlit as st
import time
from pathlib import Path
import base64
import pandas as pd
import plotly.express as px

from src.csv_processor import load_csv, profile_dataframe, process_mapped_data, CSVProcessorError
from src.ml_pipeline import batch_predict_csv, get_dashboard_kpis

from src.auth import require_login
from src.ml_pipeline import predict_single_customer
from src.llm_agent import generate_action_response

def stream_text(text):
    """Takes fully generated text and visually streams it word-by-word."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03) # Speed of the typing effect
import base64
from pathlib import Path

def set_batch_background(image_path):
    with open(image_path, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>

        .batch-hero {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;

            min-height: 300px;
            border-radius: 20px;

            padding: 40px;
            margin-bottom: 25px;

            display: flex;
            flex-direction: column;
            justify-content: center;

            box-shadow: 0 8px 30px rgba(0,0,0,0.2);
        }}

        .batch-title {{
            color: white;
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 10px;
        }}

        .batch-subtitle {{
            color: white;
            font-size: 18px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )        

def scroll_to_bottom():
    st.html("""
        <script>
            (function() {
                function scrollEl(el, tag) {
                    if (!el) { console.log('[scroll-debug] ' + tag + ': element not found'); return; }
                    console.log('[scroll-debug] ' + tag + ': scrollHeight=' + el.scrollHeight + ' clientHeight=' + el.clientHeight + ' scrollTop(before)=' + el.scrollTop);
                    try {
                        el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
                        console.log('[scroll-debug] ' + tag + ': scrollTo called, scrollTop(after)=' + el.scrollTop);
                    } catch (e) {
                        console.log('[scroll-debug] ' + tag + ': scrollTo threw, falling back. Error: ' + e);
                        el.scrollTop = el.scrollHeight;
                    }
                }

                function findScrollable() {
                    var container = document.querySelector('.st-key-campaign_chat_box');
                    console.log('[scroll-debug] container found: ' + (container ? 'yes' : 'NO'));
                    if (!container) return null;
                    if (container.scrollHeight > container.clientHeight) {
                        console.log('[scroll-debug] container itself is scrollable');
                        return container;
                    }
                    var descendants = container.querySelectorAll('*');
                    console.log('[scroll-debug] checking ' + descendants.length + ' descendants for overflow');
                    for (var i = 0; i < descendants.length; i++) {
                        if (descendants[i].scrollHeight > descendants[i].clientHeight) {
                            console.log('[scroll-debug] found scrollable descendant: ' + descendants[i].tagName + '.' + descendants[i].className);
                            return descendants[i];
                        }
                    }
                    console.log('[scroll-debug] no scrollable descendant found, using container as fallback');
                    return container;
                }

                console.log('[scroll-debug] scroll_to_bottom() invoked at ' + Date.now());
                requestAnimationFrame(function() {
                    requestAnimationFrame(function() {
                        scrollEl(findScrollable(), 'main-call');
                    });
                });
            })();
        </script>
    """, unsafe_allow_javascript=True)

def preserve_scroll():
    st.html("""
        <script>
            (function() {
                var mainEl = document.querySelector('section[data-testid="stMain"]');
                if (mainEl) {
                    var saved = sessionStorage.getItem('scrollPos');
                    if (saved !== null) { mainEl.scrollTop = parseInt(saved); }
                    mainEl.addEventListener('scroll', function() {
                        sessionStorage.setItem('scrollPos', mainEl.scrollTop);
                    });
                }
            })();
        </script>
    """, unsafe_allow_javascript=True)

preserve_scroll()

# ==========================================
# 1. PAGE SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Marketing Intelligence Platform",
    page_icon="🎯",
    layout="wide"
)


# ---------------------------------------

require_login()

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_segment" not in st.session_state:
    st.session_state.current_segment = None
if "top_category" not in st.session_state:
    st.session_state.top_category = None
if "stream_latest" not in st.session_state:
    st.session_state.stream_latest = False
if "full_screen" not in st.session_state:
    st.session_state.full_screen = False
if "scroll_pending" not in st.session_state:
    st.session_state.scroll_pending = False

# Load external CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    st.markdown("""
<style>

div.stButton > button:first-child{
    width:100%;
    height:60px;
    border:none;
    border-radius:14px;
    background:linear-gradient(
        135deg,
        #2563eb,
        #1d4ed8
    );
    color:white;
    font-size:18px;
    font-weight:700;
}

div.stButton > button:first-child:hover{
background:linear-gradient(135deg,#3b82f6,#2563eb);}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HEADER & NAVIGATION
# ==========================================
if not st.session_state.full_screen:
    
    # 1. Base64 Encode Logo
    image_path = Path(__file__).parent / "assets" / "logo.jpeg"
    
    try:
        image_data = base64.b64encode(image_path.read_bytes()).decode()
    except FileNotFoundError:
        # Failsafe just in case the image path is slightly off during testing
        image_data = ""

    # 2. Header Columns (Logo | Title | Logout)
    header_col1, header_col2, header_col3 = st.columns([0.06, 0.84, 0.10])

    with header_col1:
        if image_data:
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
            <h1 style="margin-top: 0; padding-top: 0;" class="ai-title">
                MARKETRON
            </h1>
            <h3 style="margin-top: -10px; color: #64748b;" class="ai-subtitle">
                “Segment Smarter. Market Better.”
            </h3>
            """,
            unsafe_allow_html=True
        )
        
    with header_col3:
        # st.markdown("<br>", unsafe_allow_html=True) # Pushes button down slightly to align with text
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # 3. Description & Divider
    st.markdown(
        """
        <div class="project-intro">
            <h2 class="project-title">
                Customer Segmentation & Personalized Marketing Intelligence
            </h2>
            <p class="project-description">
                Predict customer segments based on behavioral data to unlock
                personalized insights and determine the exact marketing approach
                for each specific group.
            </p>
        </div>
        <hr style="border: 0; border-top: 1px solid #E2E8F0; margin-bottom: 30px;">
        """,
        unsafe_allow_html=True
    ) 

# ==========================================
# NEW BLOCK: SIDEBAR NAVIGATION
# ==========================================

# 1. Initialize the default page on first load
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Single Customer Analysis"

with st.sidebar:
    st.markdown("### Navigation")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Dynamic Button Styling (Highlights the active page)
    btn_type_single = "primary" if st.session_state.app_mode == "Single Customer Analysis" else "secondary"
    btn_type_batch = "primary" if st.session_state.app_mode == "Batch CSV Processor" else "secondary"
    
    # 3. The Clickable Text Links (Buttons)
    if st.button("Single Customer Analysis", type=btn_type_single, use_container_width=True):
        st.session_state.app_mode = "Single Customer Analysis"
        st.rerun()
        
    if st.button("Batch CSV Processor", type=btn_type_batch, use_container_width=True):
        st.session_state.app_mode = "Batch CSV Processor"
        st.rerun()
        
    st.markdown("---")


# ==========================================
# APP ROUTING ENGINE
# ==========================================

if st.session_state.app_mode == "Single Customer Analysis":
    # ==========================================
    # 3. BLOCK 1: INPUT BANNER
    # ==========================================
    predict_btn = False
    if not st.session_state.full_screen:
        with st.container(key="cohort_banner"):
            banner_left, banner_right = st.columns([1.2, 1], gap="large")

            with banner_left:
                st.markdown("<h3 style='color: #38BDF8; margin-top: -10px;'>Segment Profile Input</h3>", unsafe_allow_html=True)
                st.write("Provide the **average engagement metrics** for this customer segment:")
            
                with st.form("customer_input_form"):
                    top_cat = st.text_input(
                        "Primary Product Category",
                        placeholder="e.g., Wireless Earbuds, Coffee Beans"
                    )
                
                    c1, c2 = st.columns(2)
                    with c1:
                        recency = st.number_input("Average Recency (Days)", min_value=0, max_value=1000, value=25)
                        monetary = st.number_input("Average Total Spend", min_value=1000.0, max_value=100000.0, value=1000.0, step=500.0)
                    with c2:
                        frequency = st.number_input("Average Frequency (Orders)", min_value=1, max_value=200, value=3)
                        # Swapped out Review Score for Price Sensitivity
                        price_sensitivity = st.selectbox(
                            "Price Sensitivity",
                            options=["Full Price Consumers", "Bargain Hunters", "Seasonal Shoppers"]
                        )
                    
                    predict_btn = st.form_submit_button("🔍 Analyze Segment Metrics",key="analyze_btn", use_container_width=True)

            with banner_right:
                st.markdown("""
                    <div style="padding-top: 20px;">
                        <h1 style="color: #38BDF8; margin-bottom: 15px; font-family: sans-serif; letter-spacing: 3px;">Segment-Level Intelligence</h1>
                        <p style="font-size: 1.05em; line-height: 1.6; color: #CBD5E1; margin-bottom: 20px;">
                            Provide average customer behavior metrics to instantly identify their segment and get an AI-crafted marketing approach built specifically for that group.
                        </p>
                    </div>
                """, unsafe_allow_html=True) 

    # ==========================================
    # 4. BLOCK 2: CENTRILIZED PROCESSING & CHAT 
    # ==========================================

    # 1. Centered Loading Animation 
    if predict_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.status("🧠 Processing Segment Intelligence...", expanded=True) as status:
            st.write("Classifying segment via ML Matrix...")
            time.sleep(0.4) 
            
            try:
                # 1. Removed review_score; passing pure RFM to the ML backend
                pred = predict_single_customer(recency, frequency, monetary)
                
                if "Champions" in pred["segment_name"]: badge_color = "green"
                elif "Risk" in pred["segment_name"]: badge_color = "red"
                elif "Churned" in pred["segment_name"]: badge_color = "gray"
                else: badge_color = "blue"

                st.session_state.current_segment = pred["segment_name"]
                st.session_state.top_category = top_cat if top_cat.strip() != "" else "General Merchandise"
                
                # 2. Saving the new dropdown value to session state for LLM injection
                st.session_state.price_sensitivity = price_sensitivity
                
                st.session_state.segment_desc = pred["description"]
                st.session_state.badge_color = badge_color
                st.session_state.chat_history = [] 
                
                st.write("Crafting your AI-powered campaign approach...")
                initial_prompt = "Provide a brief, high-level overview of exactly how we should market to this specific segment."
                _, st.session_state.chat_history = generate_action_response(
                    st.session_state.chat_history, st.session_state.current_segment, st.session_state.top_category, st.session_state.price_sensitivity,initial_prompt
                )
                
                # HIDDEN FLAG: Hide the automated initial prompt from the UI
                st.session_state.chat_history[-2]["hidden"] = True
                
                st.session_state.stream_latest = True 
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="Analysis Failed", state="error", expanded=False)
                st.error(f"Backend Error: {e}")  

    # 2. Full Width Content Layout
    if st.session_state.current_segment:
        
        st.markdown("<br>", unsafe_allow_html=True)

        # ---> HIDE THESE BLOCKS IN FULL SCREEN <---
        if not st.session_state.full_screen:
            st.markdown(f"""
                <div style="background-color: #ECFDF5; border: 1px solid #6EE7B7; border-radius: 10px; padding: 16px 20px; margin-bottom: 10px;">
                    <span style="color: #065F46; font-size: 0.95em; font-weight: 500;">Segment Match</span><br>
                    <span style="color: #047857; font-size: 1.8em; font-weight: 800; letter-spacing: 0.5px;">{st.session_state.current_segment}</span>
                </div>
            """, unsafe_allow_html=True)
        
            st.markdown(f"""
                <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 14px 20px; margin-bottom: 10px;">
                    <span style="color: #1E40AF; font-weight: 600;">Category:</span>
                    <code>{st.session_state.top_category}</code> | {st.session_state.segment_desc}
                </div>
            """, unsafe_allow_html=True)
        
            st.markdown("---")

        # Wrap buttons in a scoped container to color-code them individually
        with st.container(key="action_container"):
            
            # ⛶ FULL SCREEN NAV BAR
            nav_c1, nav_c2 = st.columns([5, 1])
            with nav_c1:
                if st.session_state.full_screen:
                    if st.button("⬅️ Back to Dashboard",key="back_btn" ,use_container_width=False):
                        st.session_state.full_screen = False
                        st.rerun()
            with nav_c2:
                if not st.session_state.full_screen:
                    if st.button("⛶ Full Screen",key="fullscreen_btn", use_container_width=True):
                        st.session_state.full_screen = True
                        st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # ACTION BUTTONS (NOW ACTING AS STATE TRIGGERS)
            btn_c1, btn_c2, btn_c3 = st.columns(3)
            
            with btn_c1:
                if st.button("📧 Email Draft", key="email_btn", use_container_width=True):
                    st.session_state.pending_prompt = "Write a high-converting marketing email template (with a subject line) for this segment. Give them a compelling reason to buy again today."
                    st.session_state.pending_action = "Drafting Email Campaign..."
                    st.rerun()
                
                st.markdown("""
                    <div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 12px; text-align: center; margin-top: 5px;">
                        <span style="color: #991B1B; font-size: 0.85em; font-weight: 500;">Direct-response template for immediate conversions.</span>
                    </div>
                """, unsafe_allow_html=True)
                        
            with btn_c2:
                if st.button("📢 Social Ad Copy", key="ad_btn", use_container_width=True):
                    st.session_state.pending_prompt = "Draft short, punchy Facebook/Instagram Ad copy for this segment. Include a Headline, Body Text, and CTA."
                    st.session_state.pending_action = "Drafting Ad Campaign..."
                    st.rerun()
                
                st.markdown("""
                    <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 12px; text-align: center; margin-top: 5px;">
                        <span style="color: #92400E; font-size: 0.85em; font-weight: 500;">Scroll-stopping social media ad creative.</span>
                    </div>
                """, unsafe_allow_html=True)
                        
            with btn_c3:
                if st.button("🎯 Retention Strategy", key="strategy_btn", use_container_width=True):
                    st.session_state.pending_prompt = "Provide a detailed, bulleted 3-step retention strategy and follow-up sequence for this segment."
                    st.session_state.pending_action = "Building Strategy..."
                    st.rerun()
                
                st.markdown("""
                    <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 12px; text-align: center; margin-top: 5px;">
                        <span style="color: #1E40AF; font-size: 0.85em; font-weight: 500;">Step-by-step engagement and retention plan.</span>
                    </div>
                """, unsafe_allow_html=True)
        
        # ==========================================
        # 5. INTELLIGENT CHAT ENGINE
        # ==========================================
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Chat box always has a fixed scroll height — taller in fullscreen since it's the main content there
        chat_height = 650 if st.session_state.full_screen else 500
        chat_box = st.container(height=chat_height, key="campaign_chat_box")
        
        with chat_box:
            if len(st.session_state.chat_history) == 0:
                st.caption("Click a campaign button above or use the custom prompt menu to start drafting.")
            
            # Render past history
            for msg in st.session_state.chat_history:
                if msg["role"] == "system" or msg.get("hidden", False):
                    continue
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if st.session_state.scroll_pending:
            scroll_to_bottom()
            st.session_state.scroll_pending = False

            # PROMPT INPUT (Pins to the bottom of the screen, outside the container)
        chat_prompt = st.chat_input("Refine this campaign (e.g., 'Make it more urgent', 'Add a 20% discount')...")

        pending_prompt = st.session_state.get("pending_prompt")

        # 4. EXECUTION & IN-LINE LOADING
        if chat_prompt or pending_prompt:
            prompt = chat_prompt if chat_prompt else pending_prompt
            action_text = st.session_state.get("pending_action", "Refining Campaign...")

            # Button prompts stay hidden, typed prompts render in the UI
            is_hidden = True if pending_prompt else False
            st.session_state.chat_history.append({"role": "user", "content": prompt, "hidden": is_hidden})

            # 🛠️ THE FIX: Force the generation UI to render INSIDE the scrollable chat box
            with chat_box:
                
                if not is_hidden:
                    with st.chat_message("user"):
                        st.markdown(prompt)

                scroll_to_bottom()

                # Render the loading animation INSIDE the AI's chat bubble
                with st.chat_message("assistant"):
                    with st.status(action_text, expanded=True) as status:
                        _, st.session_state.chat_history = generate_action_response(
                            st.session_state.chat_history, st.session_state.current_segment, st.session_state.top_category, 
                            st.session_state.get("price_sensitivity", "Full Price Consumers"),
                            prompt
                        )

                        status.update(label="Complete!", state="complete", expanded=False)

                    st.write_stream(stream_text(st.session_state.chat_history[-1]["content"]))

            st.session_state.scroll_pending = True

            # Clean the triggers and finalize the render
            if "pending_prompt" in st.session_state:
                del st.session_state["pending_prompt"]
            if "pending_action" in st.session_state:
                del st.session_state["pending_action"]
            st.rerun()  
    pass
    
elif st.session_state.app_mode == "Batch CSV Processor":
     
      # ---> APPLY BACKGROUND ONLY HERE <---
    def set_background():  
     bg_img = Path(r"C:\Users\sivan\OneDrive\Desktop\Marketing_Intelligence_Platform\assets\background.jpg")
     if bg_img.exists():
        set_background(str(bg_img))    
    # ------------------------------------
  
    st.markdown("## 📂 Batch Customer Segmentation")
    st.write("Upload a CSV file to analyze thousands of customers at once.")

    # SMALL UPLOAD BUTTON LEFT SIDE
uploaded_file = st.file_uploader(
        "",
        type=["csv"],
        label_visibility="collapsed"
    )
if uploaded_file :

        try:

            raw_df = load_csv(uploaded_file)
            profile = profile_dataframe(raw_df)

            csv_headers = profile["column_names"]

            st.markdown(f"""
            <div style="
                background:#ecfdf5;
                border:1px solid #6ee7b7;
                padding:16px;
                border-radius:12px;
                margin-top:10px;
                margin-bottom:20px;
            ">
                <h4 style="color:#047857;margin:0;background-color:black!important">
                    ✅ File loaded successfully!
                </h4>
                <p style="margin:0;color:#065f46;">
                    {profile['rows']:,} rows •
                    {profile['columns']} columns
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
<style>
.data-config-title{font-size:38px;font-weight:800;color:#67e8f9;margin-bottom:15px;}
</style>
""", unsafe_allow_html=True)


            st.markdown('<div class="data-config-title">## ⚙️ Data Configuration</div>',unsafe_allow_html=True)

            mapping_mode = st.radio(
                "What kind of data uploading?",
                
                ["direct_rfm", "raw_transactions"],
                horizontal=True,
                format_func=lambda x:
                    "📊 Pre-Calculated RFM(Recency,Frequency,Monetry)"
                    if x == "direct_rfm"
                    else "🛒 Raw Transaction Logs(Customer ID,Dates,Spend)"
            )
            

            st.markdown("---")

            st.markdown("## 🗺️ Map Your Columns")

            column_map = {}

            if mapping_mode == "direct_rfm":

                c1, c2, c3 = st.columns(3)

                with c1:
                    column_map["recency"] = st.selectbox(
                        "📅 Recency Column",
                        csv_headers
                    )

                with c2:
                    column_map["frequency"] = st.selectbox(
                        "🔄 Frequency Column",
                        csv_headers
                    )

                with c3:
                    column_map["monetary"] = st.selectbox(
                        "💰 Monetary Column",
                        csv_headers
                    )

            else:

                c1, c2 = st.columns(2)

                with c1:

                    column_map["customer_id"] = st.selectbox(
                        "👤 Customer ID",
                        csv_headers
                    )

                    column_map["order_date"] = st.selectbox(
                        "📅 Order Date",
                        csv_headers
                    )

                with c2:

                    column_map["order_id"] = st.selectbox(
                        "🧾 Order ID",
                        csv_headers
                    )

                    column_map["spend"] = st.selectbox(
                        "₹ Spend / Price",
                        csv_headers
                    )

            st.markdown("""
            <div style="
                background:#ecfdf5;
                border:1px solid #22c55e;
                padding:18px;
                border-radius:12px;
                margin-top:20px;
                margin-bottom:20px;
            ">
                <h4 style="margin:0;color:#166534;">
                    ✅ Data structure detected
                </h4>
                <p style="margin-top:5px;color:#166534;">
                    Your CSV is ready for MARKETRON's
                    segmentation pipeline.
                </p>
            </div>
            """, unsafe_allow_html=True)

            process_btn = st.button(
                "🚀 Process Batch Data",
                use_container_width=True
            )

            if process_btn:

                with st.spinner(
                    "Classifying customer segments..."
                ):

                    try:

                        mapped_df = process_mapped_data(
                            raw_df,
                            mapping_mode,
                            column_map
                        )

                        results_df = batch_predict_csv(
                            mapped_df
                        )

                        kpis = get_dashboard_kpis(
                            results_df
                        )

                        st.session_state.batch_results = results_df
                        st.session_state.batch_kpis = kpis
                        st.session_state.batch_processed = True

                        st.rerun()

                    except CSVProcessorError as e:
                        st.error(
                            f"⚠️ Data Ingestion Error: {e}"
                        )

                    except ValueError as e:
                        st.error(
                            f"⚠️ Data Validation Error: {e}"
                        )

                    except Exception as e:
                        st.error(
                            f"⚠️ Unexpected Error: {e}"
                        )

        except CSVProcessorError as e:
            st.error(
                f"⚠️ File Load Error: {e}"
            )

st.markdown("</div>", unsafe_allow_html=True)


# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

/* Remove full width uploader */
[data-testid="stFileUploader"]{
    width:280px !important;
}

/* Upload button */
[data-testid="stFileUploader"] section button{
    width:140px !important;
    height:42px !important;
    border-radius:12px !important;
    font-size:16px !important;
    font-weight:600 !important;
}

/* Drop area */
[data-testid="stFileUploaderDropzone"]{
    min-height:90px !important;
    width:280px !important;
    border-radius:12px !important;
}

</style>
""", unsafe_allow_html=True)

    # ============================================================
    # BATCH DASHBOARD (Premium Visualizations)
    # ============================================================
if st.session_state.get("batch_processed"):
        st.markdown("---")
        st.markdown("## 📊 Batch Segmentation Results")
        
        results_df = st.session_state.batch_results
        kpis = st.session_state.batch_kpis
        
        # 1. TOP-LEVEL KPI CARDS
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Customers", f"{kpis['total_customers']:,}")
        c2.metric("Average Spend", f"₹{kpis['average_monetary']:,.2f}")
        c3.metric("Total Segments", len(kpis["segment_distribution"]))
        
        st.markdown("<br>", unsafe_allow_html=True)
        
          # ============================================================
              # 2. FULL-WIDTH VISUALIZATIONS
# ============================================================

# ------------------------------------------------------------
# 2A. SEGMENT DISTRIBUTION - CENTERED DONUT
# ------------------------------------------------------------
        st.markdown(
    """
    <div style="text-align:center; margin-top:10px;">
        <h3 style="color:#38BDF8; margin-bottom:4px;">
            Segment Distribution
        </h3>
        <p style="color:#94A3B8; margin-top:0;">
            Percentage of your total customer base.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

        segment_counts = (
    results_df["Segment"]
    .value_counts()
    .rename_axis("Segment")
    .reset_index(name="Count")
)

        fig_pie = px.pie(
    segment_counts,
    values="Count",
    names="Segment",
    hole=0.55,
    color="Segment"
)

        fig_pie.update_traces(
    textinfo="percent",
    textposition="inside",
    hovertemplate=(
        "<b>%{label}</b><br>"
        "Customers: %{value:,}<br>"
        "Percentage: %{percent}<extra></extra>"
    )
)

        fig_pie.update_layout(
    height=520,
    
    legend=dict(
        orientation="v",
        yanchor="middle",
        y=0.5,
        xanchor="left",
        x=0.72,
        font=dict(size=14)
    ),
    margin=dict(l=50,r=180,t=60,b=40)

)

        st.plotly_chart(
    fig_pie,
    use_container_width=True
)

        st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------------------------------------
# 2B. CUSTOMER COUNT PER SEGMENT
#     FULL-WIDTH VERTICAL BAR CHART
# ------------------------------------------------------------
        st.markdown(
    """
    <div style="text-align:center;">
        <h3 style="color:#38BDF8; margin-bottom:4px;">
            Customer Count per Segment
        </h3>
        <p style="color:#94A3B8; margin-top:0;">
            Total volume of customers in each cohort.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

        fig_bar_count = px.bar(
    segment_counts,
    x="Segment",
    y="Count",
    color="Segment",
    text="Count"
)

        fig_bar_count.update_traces(
    texttemplate="%{text:,}",
    textposition="outside",
    hovertemplate=(
        "<b>%{x}</b><br>"
        "Customers: %{y:,}<extra></extra>"
    )
)

        fig_bar_count.update_layout(
    height=500,
    margin=dict(
        t=30,
        b=80,
        l=70,
        r=40
    ),
    xaxis_title="",
    yaxis_title="Number of Customers",
    showlegend=False
)

        st.plotly_chart(
    fig_bar_count,
    use_container_width=True
)

        st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------------------------------------
# 2C. AVERAGE SPEND PER SEGMENT
#     FULL-WIDTH VERTICAL BAR CHART
# ------------------------------------------------------------
        st.markdown(
    """
    <div style="text-align:center;">
        <h3 style="color:#38BDF8; margin-bottom:4px;">
            Average Spend per Segment
        </h3>
        <p style="color:#94A3B8; margin-top:0;">
            Monetary value generated by each cohort.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

        avg_spend_df = (
    results_df
    .groupby("Segment", as_index=False)["Monetary"]
    .mean()
    .rename(
        columns={
            "Monetary": "AverageSpend"
        }
    )
)

        fig_bar_spend = px.bar(
    avg_spend_df,
    x="Segment",
    y="AverageSpend",
    color="Segment",
    text="AverageSpend"
)

        fig_bar_spend.update_traces(
    texttemplate="₹%{text:,.0f}",
    textposition="outside",
    hovertemplate=(
        "<b>%{x}</b><br>"
        "Average Spend: ₹%{y:,.2f}<extra></extra>"
    )
)

        fig_bar_spend.update_layout(
    height=500,
    margin=dict(
        t=30,
        b=80,
        l=80,
        r=40
    ),
    xaxis_title="",
    yaxis_title="Avg Spend (₹)",
    showlegend=False
)

        st.plotly_chart(
    fig_bar_spend,
    use_container_width=True
)

        st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------------------------------------
# 2D. 3D CUSTOMER UNIVERSE
#     FULL WIDTH
# ------------------------------------------------------------
        st.markdown(
    """
    <div style="text-align:center;">
        <h3 style="color:#38BDF8; margin-bottom:4px;">
            3D Customer Universe
        </h3>
        <p style="color:#94A3B8; margin-top:0;">
            Interactive map of your customer base (sampled for speed).
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sample customers to keep the 3D visualization responsive
        sample_df = results_df.sample(
    min(1000, len(results_df)),
    random_state=42
)

        fig_3d = px.scatter_3d(
    sample_df,
    x="Recency",
    y="Frequency",
    z="Monetary",
    color="Segment",
    opacity=0.7,
    size_max=10
)

        fig_3d.update_layout(
    height=650,
    margin=dict(
        t=10,
        b=10,
        l=10,
        r=10
    )
)

        st.plotly_chart(
    fig_3d,
    use_container_width=True
)
        # 4. FULL INTERACTIVE DATABASE
        st.markdown("### 📋 Customer Database")
        st.write("Your original data, now enhanced with ML segment predictions. You can sort and filter this table directly.")
        st.dataframe(results_df, use_container_width=True, height=250)
        
        csv_export = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Segmented CSV",
            data=csv_export,
            file_name="ml_segmented_customers.csv",
            mime="text/csv"
        )

        st.markdown("---")
        
        # 5. AI CAMPAIGN GENERATOR
        st.markdown("### 🤖 Generate Targeted Marketing Strategy")
        st.write("Select a customer segment to instantly generate a hyper-personalized marketing approach using our AI agent.")
        
        available_segments = sorted(results_df["Segment"].unique())
        
        gen_col1, gen_col2 = st.columns([2, 1])
        
        with gen_col1:
            target_segment = st.selectbox(
                "Select Target Segment:", 
                options=available_segments,
                label_visibility="collapsed"
            )
            
        with gen_col2:
            if st.button("🚀 Generate Approach", use_container_width=True):
                st.session_state.target_generation_segment = target_segment
                st.session_state.trigger_ai_generation = True
                
        if st.session_state.get("trigger_ai_generation"):
            st.info(f"Connecting to AI Agent to generate strategy for **{st.session_state.target_generation_segment}**... (LLM logic goes here!)")
            
        st.markdown("---")
        if st.button("🔄 Upload New File"):
            st.session_state.batch_processed = False
            st.rerun()
pass
