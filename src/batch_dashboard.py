import streamlit as st
import pandas as pd
from src.csv_processor import load_csv, profile_dataframe, process_mapped_data, CSVProcessorError
from src.ml_pipeline import batch_predict_csv, get_dashboard_kpis

st.markdown("## 📂 Batch Customer Segmentation")
st.write("Upload a CSV file to analyze thousands of customers at once.")

uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type=["csv"])

if uploaded_file:
    # 1. LOAD & PROFILE
    try:
        raw_df = load_csv(uploaded_file)
        profile = profile_dataframe(raw_df)
        csv_headers = profile["column_names"]
        
        st.success(f"✅ File loaded successfully! ({profile['rows']} rows, {profile['columns']} columns)")
        
        # 2. SELECT PROCESSING MODE
        st.markdown("### ⚙️ Data Configuration")
        mapping_mode = st.radio(
            "What kind of data are you uploading?",
            options=["direct_rfm", "raw_transactions"],
            format_func=lambda x: "📈 Pre-Calculated RFM (Recency, Frequency, Monetary)" if x == "direct_rfm" else "🛒 Raw Transaction Logs (Customer ID, Dates, Spend)"
        )
        
        # 3. STRICT MANUAL COLUMN MAPPING
        st.markdown("### 🗺️ Map Your Columns")
        column_map = {}
        
        if mapping_mode == "direct_rfm":
            # PATH A: Direct RFM Mapping
            c1, c2, c3 = st.columns(3)
            with c1:
                column_map["recency"] = st.selectbox("Recency Column", options=csv_headers)
            with c2:
                column_map["frequency"] = st.selectbox("Frequency Column", options=csv_headers)
            with c3:
                column_map["monetary"] = st.selectbox("Monetary Column", options=csv_headers)
        
        else:
            # PATH B: Raw Transactions Mapping
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                column_map["customer_id"] = st.selectbox("Customer ID", options=csv_headers)
            with c2:
                column_map["order_date"] = st.selectbox("Order Date", options=csv_headers)
            with c3:
                column_map["order_id"] = st.selectbox("Order ID", options=csv_headers)
            with c4:
                column_map["spend"] = st.selectbox("Spend/Price", options=csv_headers)
        
        # 4. EXECUTION & ERROR HANDLING
        if st.button("🚀 Process Batch Data", use_container_width=True):
            with st.spinner("Classifying segments..."):
                try:
                    # Execute Router
                    mapped_df = process_mapped_data(raw_df, mapping_mode, column_map)
                    
                    # Execute ML Pipeline
                    results_df = batch_predict_csv(mapped_df)
                    
                    # Generate Dashboard KPIs
                    kpis = get_dashboard_kpis(results_df)
                    
                    # Save results to session state
                    st.session_state.batch_results = results_df
                    st.session_state.batch_kpis = kpis
                    st.session_state.batch_processed = True
                    
                    st.rerun() 
                    
                except CSVProcessorError as e:
                    st.error(f"⚠️ **Data Ingestion Error:** {e}")
                except ValueError as e:
                    st.error(f"⚠️ **Data Validation Error:** {e}")
                except Exception as e:
                    st.error(f"⚠️ **An unexpected error occurred:** {e}")
                    
    except CSVProcessorError as e:
        st.error(f"⚠️ **File Load Error:** {e}")

# ============================================================
# BATCH DASHBOARD (Premium Visualizations)
# ============================================================
import plotly.express as px

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
    
    # 2. PIE & BAR CHARTS (Row 1)
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### Segment Distribution")
        st.write("Percentage of your total customer base.")
        fig_pie = px.pie(results_df, names="Segment", hole=0.4, color="Segment")
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with chart_col2:
        st.markdown("#### Customer Count per Segment")
        st.write("Total volume of customers in each cohort.")
        fig_bar_count = px.histogram(results_df, x="Segment", color="Segment")
        fig_bar_count.update_layout(xaxis_title="", yaxis_title="Number of Customers", showlegend=False)
        st.plotly_chart(fig_bar_count, use_container_width=True)

    # 3. SPEND & 3D SCATTER (Row 2)
    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        st.markdown("#### Average Spend per Segment")
        st.write("Monetary value generated by each cohort.")
        avg_spend_df = results_df.groupby("Segment")["Monetary"].mean().reset_index()
        fig_bar_spend = px.bar(avg_spend_df, x="Segment", y="Monetary", color="Segment")
        fig_bar_spend.update_layout(xaxis_title="", yaxis_title="Avg Spend (₹)", showlegend=False)
        st.plotly_chart(fig_bar_spend, use_container_width=True)
        
    with chart_col4:
        st.markdown("#### 3D Customer Universe")
        st.write("Interactive map of your customer base (Sampled for speed).")
        # Sample data so the browser doesn't lag on huge CSVs
        sample_df = results_df.sample(min(1000, len(results_df)))
        fig_3d = px.scatter_3d(
            sample_df, x="Recency", y="Frequency", z="Monetary", 
            color="Segment", opacity=0.7, size_max=10
        )
        fig_3d.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_3d, use_container_width=True)

# ============================================================
# BLOCK 4.2: DATA TABLE & AI CAMPAIGN GENERATOR
# ============================================================
    
    st.markdown("---")
    
    # 4. FULL INTERACTIVE DATABASE
    st.markdown("### 📋 Customer Database")
    st.write("Your original data, now enhanced with ML segment predictions. You can sort and filter this table directly.")
    st.dataframe(results_df, use_container_width=True, height=250)
    
    # Download button for the raw data
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
    
    # Dynamic list of whatever segments the ML model found
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
            
    # This is a placeholder for where the LLM response will render
    if st.session_state.get("trigger_ai_generation"):
        st.info(f"Connecting to AI Agent to generate strategy for **{st.session_state.target_generation_segment}**... (LLM logic goes here!)")
        
    st.markdown("---")
    if st.button("🔄 Upload New File"):
        st.session_state.batch_processed = False
        st.rerun()