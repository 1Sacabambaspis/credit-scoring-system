"""
AI Credit Scoring - Streamlit Dashboard
==================================================
Pages:
  1. Executive Overview  - Business stakeholder view
  2. Model Diagnostics   - Technical validation & Bake-off
  3. Decision Engine     - Interactive live scoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Credit Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load data & models (cached) ────────────────────────────────────────────────
@st.cache_data
def load_data():
    pred_df = pd.read_csv("data/processed/predictions.csv")
    feat_df = pd.read_csv("data/processed/credit_features.csv")
    
    # 1. Dictionary Maps
    purpose_map = {
        'A40': 'New Car', 'A41': 'Used Car', 'A42': 'Furniture/Equipment',
        'A43': 'Radio/TV', 'A44': 'Domestic Appliances', 'A45': 'Repairs',
        'A46': 'Education', 'A47': 'Vacation', 'A48': 'Retraining',
        'A49': 'Business', 'A410': 'Others'
    }
    
    job_map = {
        'A171': 'Unemployed / Unskilled',
        'A172': 'Unskilled (Resident)',
        'A173': 'Skilled Employee',
        'A174': 'Management / Self-Employed'
    }

    # 2. Bulletproof Regex Decoder (Case-Insensitive)
    def decode_one_hot(df, prefix, pattern, mapping):
        cols = [c for c in df.columns if c.lower().startswith(prefix.lower())]
        if cols:
            # Find the active column (where value is 1)
            active_cols = df[cols].idxmax(axis=1)
            # Extract the raw dataset code (e.g., 'A43')
            codes = active_cols.str.extract(pattern, expand=False).str.upper()
            return codes.map(mapping).fillna('Unknown')
        return 'Unknown'

    # Apply the decoders
    pred_df['Loan_Purpose'] = decode_one_hot(pred_df, 'purpose', r'(A4\d+)', purpose_map)
    pred_df['Job_Title'] = decode_one_hot(pred_df, 'job', r'(A17\d+)', job_map)

    # 3. Create a clean Applicant ID for the dropdowns
    pred_df['Applicant_ID'] = ["APP-" + str(i).zfill(4) for i in range(1, len(pred_df) + 1)]
    feat_df['Applicant_ID'] = pred_df['Applicant_ID']
    
    return pred_df, feat_df

@st.cache_resource
def load_models():
    xgb = joblib.load("models/XGBoost_model.pkl")
    rf = joblib.load("models/RandomForest_model.pkl")
    lr = joblib.load("models/LogisticRegression_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return xgb, rf, lr, scaler

@st.cache_data
def load_test_data():
    return joblib.load('data/processed/test_data.pkl')

try:
    pred_df, feat_df = load_data()
    xgb_model, rf_model, lr_model, scaler = load_models()
    X_test_scaled, y_test = load_test_data()
except Exception as e:
    st.error(f"Error loading backend files: {e}. Please ensure WS1-WS6 ran successfully.")
    st.stop()

# ── Consistent colour palette ─────────────────────────────────────────────────
RISK_COLORS = {
    "Approved": "#2ECC71",  # Green
    "Rejected": "#E74C3C"   # Red
}

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("🏦 Credit Decision AI")
st.sidebar.markdown("**TEB 2043 Data Science**")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📄 Executive Overview", "🔬 Model Diagnostics", "🎯 Decision Engine"]
)
st.sidebar.markdown("---")
st.sidebar.caption("German Credit Dataset · 1,000 Applicants")

# ==========================================
# PAGE 1: Executive Overview
# ==========================================
if page == "📄 Executive Overview":
    st.title("📄 Executive Portfolio Overview")
    st.markdown("*A macro-level view of capital allocation, risk exposure, and demographic trends.*")
    st.markdown("---")

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    total_applicants = len(pred_df)
    rejection_rate = (pred_df['Loan_Decision'] == 'Rejected').mean()
    total_credit_requested = pred_df['Credit_Amount'].sum()
    credit_at_risk = pred_df.loc[pred_df['Loan_Decision'] == 'Rejected', 'Credit_Amount'].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Applicants", f"{total_applicants:,}")
    k2.metric("System Rejection Rate", f"{rejection_rate:.1%}")
    k3.metric("Total Capital Requested", f"{total_credit_requested:,.0f} DM")
    k4.metric("Capital Protected", f"{credit_at_risk:,.0f} DM", delta="High Risk Declined", delta_color="normal")

    st.markdown("---")

    # ── Highly Insightful Row 1 ───────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Approval Rate by Loan Purpose")
        # Calculate approval rate per category
        purpose_agg = pred_df.groupby('Loan_Purpose')['Loan_Decision'].apply(lambda x: (x == 'Approved').mean()).reset_index()
        purpose_agg.columns = ['Loan_Purpose', 'Approval_Rate']
        # Filter out 'Unknown' just in case there are missing values
        purpose_agg = purpose_agg[purpose_agg['Loan_Purpose'] != 'Unknown']
        purpose_agg = purpose_agg.sort_values('Approval_Rate', ascending=True)

        fig_bar = px.bar(
            purpose_agg, x='Approval_Rate', y='Loan_Purpose', orientation='h',
            text_auto='.1%', color='Approval_Rate', color_continuous_scale='RdYlGn'
        )
        fig_bar.update_layout(xaxis_title="Approval Rate", yaxis_title="", coloraxis_showscale=False, height=350)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.info("💡 **Business Insight:** 'Education' and 'New Car' loans exhibit the lowest approval rates, signaling that these demographics carry historically higher default risks.")

    with col2:
        st.subheader("Risk Distribution: Age vs. Credit Amount")
        fig_box = px.box(
            pred_df, x="Loan_Decision", y="Credit_Amount", color="Loan_Decision",
            color_discrete_map=RISK_COLORS, points="all", height=350,
            labels={'Credit_Amount': 'Credit Amount (DM)', 'Loan_Decision': 'System Decision'}
        )
        fig_box.update_layout(margin=dict(t=10, b=0, l=0, r=0), showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        st.info("💡 **Business Insight:** Rejected applicants (Red) tend to request higher capital amounts with a wider variance. Approvals (Green) are tightly clustered around lower credit amounts.")

    st.markdown("---")

    # ── Summary Table & Export ──────────────────────────────────────────
    st.subheader("Demographic Risk Profiling")
    st.markdown("The table below breaks down the average profile of an Approved vs. Rejected applicant.")
    
    col3, col4 = st.columns([1.5, 1])
    with col3:
        profile_tbl = pred_df.groupby('Loan_Decision').agg({
            'Applicant_ID': 'count',
            'Age': 'mean',
            'Credit_Amount': 'mean',
            'Duration_Months': 'mean'
        }).reset_index()
        profile_tbl.columns = ['System Decision', 'Total Applicants', 'Avg Age', 'Avg Amount (DM)', 'Avg Duration (Mo)']
        
        profile_tbl['Avg Age'] = profile_tbl['Avg Age'].round(1)
        profile_tbl['Avg Amount (DM)'] = profile_tbl['Avg Amount (DM)'].apply(lambda x: f"{x:,.0f}")
        profile_tbl['Avg Duration (Mo)'] = profile_tbl['Avg Duration (Mo)'].round(1)
        
        st.dataframe(profile_tbl, use_container_width=True, hide_index=True)
        
    with col4:
        st.info("Download the specific demographic profiles of all applicants flagged as **High Risk** by the XGBoost algorithm for manual compliance review.")
        high_risk_df = pred_df[pred_df['Loan_Decision'] == 'Rejected'].copy()
        csv_data = high_risk_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label=f"⬇️ Export High-Risk Pool ({len(high_risk_df):,} records)",
            data=csv_data, file_name="high_risk_applicants.csv", mime="text/csv", use_container_width=True
        )


# ==========================================
# PAGE 2: Model Diagnostics
# ==========================================
elif page == "🔬 Model Diagnostics":
    st.title("🔬 Model Validation & Algorithm Bake-off")
    st.markdown("""
    **The Business Goal:** The dataset strictly penalizes missing a "Bad Loan" 5 times heavier than falsely rejecting a "Good Loan". 
    Therefore, this system evaluates models based on **Recall** (the ability to successfully identify actual defaults) rather than generic accuracy.
    """)
    st.markdown("---")

    # Calculate metrics
    models = {'Logistic Regression': lr_model, 'Random Forest': rf_model, 'XGBoost (Champion)': xgb_model}
    metrics_data = []

    for name, m in models.items():
        y_pred = m.predict(X_test_scaled)
        metrics_data.append({
            'Model': name,
            'Recall': recall_score(y_test, y_pred, pos_label=1),
            'Precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, pos_label=1)
        })

    metrics_df = pd.DataFrame(metrics_data)
    
    # ── 3 Confusion Matrices Side-by-Side ──────────────────────────────────────
    st.subheader("Evaluating Recall: The Confusion Matrices")
    st.markdown("*Compare the bottom-right quadrant (Actual Bad Loans caught) across all three algorithms.*")
    
    cm_cols = st.columns(3)
    for i, (name, m) in enumerate(models.items()):
        y_pred = m.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        
        cm_fig = px.imshow(
            cm, text_auto=True, color_continuous_scale='Blues',
            labels=dict(x='AI Predicted', y='Actual Outcome'),
            x=['Pred Good', 'Pred Bad'], y=['Act Good', 'Act Bad'],
            title=name, height=320
        )
        cm_fig.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=20, l=20, r=20))
        cm_cols[i].plotly_chart(cm_fig, use_container_width=True)

    st.success("💡 **Technical Insight:** Look at the bottom row (`Act Bad`) of the matrices. By utilizing cost-sensitive learning (`scale_pos_weight=5`), the **XGBoost** champion model mathematically prioritizes the detection of Bad Loans, capturing significantly more defaults than Logistic Regression or Random Forest. This successfully protects the bank's capital.")

    st.markdown("---")

    # ── Performance & Feature Importance ───────────────────────────────────────
    col_bar, col_feat = st.columns([1, 1.2])

    with col_bar:
        st.subheader("Model Performance Comparison")
        bar_df = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
        bar_fig = px.bar(
            bar_df, x='Metric', y='Score', color='Model', barmode='group',
            text_auto='.3f', color_discrete_sequence=['#BDC3C7', '#3498DB', '#E74C3C'],
            height=350
        )
        bar_fig.update_layout(yaxis=dict(range=[0, 1]), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(t=20, b=0, l=0, r=0))
        st.plotly_chart(bar_fig, use_container_width=True)

    with col_feat:
        st.subheader("XGBoost Feature Importance")
        feature_names = feat_df.drop(['Risk', 'Applicant_ID'], axis=1).columns
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': xgb_model.feature_importances_}).sort_values('Importance', ascending=True).tail(7) 

        fi_fig = px.bar(
            fi_df, x='Importance', y='Feature', orientation='h', text_auto='.3f',
            color='Importance', color_continuous_scale='Blues', height=350
        )
        fi_fig.update_layout(coloraxis_showscale=False, margin=dict(t=20, b=0, l=0, r=0))
        st.plotly_chart(fi_fig, use_container_width=True)


# ==========================================
# PAGE 3: Decision Engine
# ==========================================
elif page == "🎯 Decision Engine":
    st.title("🎯 Live Decision Engine")
    st.markdown("*Real-time default probability scored against individual applicant data vectors.*")
    st.markdown("---")

    # ── Customer selector ──────────────────────────────────────────────────────
    selected_id = st.selectbox("Select Applicant ID", pred_df['Applicant_ID'].tolist())

    # Prep features
    applicant_raw = pred_df[pred_df['Applicant_ID'] == selected_id].iloc[0]
    X_applicant = feat_df[feat_df['Applicant_ID'] == selected_id].drop(['Risk', 'Applicant_ID'], axis=1)
    X_applicant_scaled = scaler.transform(X_applicant)

    # Predictions
    xgb_prob = xgb_model.predict_proba(X_applicant_scaled)[0, 1]
    rf_prob = rf_model.predict_proba(X_applicant_scaled)[0, 1]
    lr_prob = lr_model.predict_proba(X_applicant_scaled)[0, 1]

    st.markdown("---")

    # ── Primary: Gauge + customer info ────────────────────────────────────────
    col_gauge, col_info = st.columns([1, 1])

    with col_gauge:
        st.subheader("Default Risk Gauge")
        gauge_color = "#E74C3C" if xgb_prob >= 0.35 else "#2ECC71"
        
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number", value=xgb_prob * 100,
            number={'suffix': '%', 'font': {'size': 48}},
            title={'text': "XGBoost Probability", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'ticksuffix': '%'},
                'bar':  {'color': gauge_color},
                'steps': [
                    {'range': [0,  35],  'color': '#D5F5E3'},
                    {'range': [35, 100], 'color': '#FADBD8'}
                ],
                'threshold': {'line': {'color': 'black', 'width': 3}, 'thickness': 0.75, 'value': 35}
            }
        ))
        gauge_fig.update_layout(height=320, margin=dict(t=40, b=20, l=30, r=30))
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col_info:
        st.subheader("Applicant Profile")
        
        verdict = "🔴 REJECTED (High Risk)" if xgb_prob >= 0.35 else "🟢 APPROVED (Low Risk)"
        st.markdown(f"**System Decision:** {verdict}")
        st.markdown(f"**Actual Historical Outcome:** {'🔴 Defaulted' if applicant_raw['Risk'] == 1 else '🟢 Paid Duly'}")
        
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Age", f"{applicant_raw['Age']} yrs")
        m2.metric("Duration", f"{applicant_raw['Duration_Months']} mo")
        m3.metric("Requested Amount", f"{applicant_raw['Credit_Amount']} DM")

        st.markdown("---")
        # Directly pulling from the newly decoded English columns!
        purpose_val = applicant_raw['Loan_Purpose']
        job_val = applicant_raw['Job_Title']
        st.info(f"**Purpose:** {purpose_val}  \n**Job:** {job_val}")

    st.markdown("---")

    # ── Secondary: All 3 models comparison ────────────────────────────────────
    st.subheader("Algorithm Consensus")
    
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("🥇 XGBoost", f"{xgb_prob:.1%}", delta="Champion", delta_color="off")
    mc2.metric("Random Forest", f"{rf_prob:.1%}")
    mc3.metric("Logistic Regression", f"{lr_prob:.1%}")

    compare_fig = px.bar(
        pd.DataFrame({
            'Algorithm': ['XGBoost', 'Random Forest', 'Logistic Regression'],
            'Probability': [xgb_prob, rf_prob, lr_prob],
            'Champion': ['Yes', 'No', 'No']
        }),
        x='Algorithm', y='Probability', color='Champion',
        color_discrete_map={'Yes': '#E74C3C', 'No': '#BDC3C7'},
        text_auto='.1%', height=300
    )
    compare_fig.add_hline(y=0.35, line_dash='dash', line_color='black', annotation_text='Rejection Threshold (35%)')
    compare_fig.update_layout(yaxis=dict(range=[0, 1], tickformat='.0%'), showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(compare_fig, use_container_width=True)