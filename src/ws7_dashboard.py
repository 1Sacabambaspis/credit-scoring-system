"""
AI Credit Scoring - Streamlit Dashboard
==================================================
Pages:
  1. Executive Overview  - Business stakeholder view
  2. Model Diagnostics   - Technical validation
  3. Decision Engine     - Interactive live scoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Credit Scoring Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load data & models (cached) ────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Load the predictions and the raw features
    pred_df = pd.read_csv("data/processed/predictions.csv")
    feat_df = pd.read_csv("data/processed/credit_features.csv")
    
    # Create a dummy Applicant ID for the dropdowns
    pred_df['Applicant_ID'] = ["APP-" + str(i).zfill(4) for i in range(1, len(pred_df) + 1)]
    feat_df['Applicant_ID'] = pred_df['Applicant_ID']
    return pred_df, feat_df

@st.cache_resource
def load_models():
    # Load the models and the scaler
    xgb = joblib.load("models/XGBoost_model.pkl")
    rf = joblib.load("models/RandomForest_model.pkl")
    lr = joblib.load("models/LogisticRegression_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return xgb, rf, lr, scaler

@st.cache_data
def load_test_data():
    return joblib.load('data/processed/test_data.pkl')

pred_df, feat_df = load_data()
xgb_model, rf_model, lr_model, scaler = load_models()
X_test_scaled, y_test = load_test_data()

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
st.sidebar.caption("German Credit Dataset · 1,000 Applicants · Tuned XGBoost")

# ==========================================
# PAGE 1: Executive Overview
# ==========================================
if page == "📄 Executive Overview":
    st.title("📄 Executive Portfolio Overview")
    st.markdown("*Business-level summary of loan distribution and risk exposure.*")
    st.markdown("---")

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    total_applicants = len(pred_df)
    rejection_rate = (pred_df['Loan_Decision'] == 'Rejected').mean()
    total_credit_requested = pred_df['Credit_Amount'].sum()
    credit_at_risk = pred_df.loc[pred_df['Loan_Decision'] == 'Rejected', 'Credit_Amount'].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Applicants", f"{total_applicants:,}")
    k2.metric("AI Rejection Rate", f"{rejection_rate:.1%}", help="Based on 35% XGBoost probability threshold.")
    k3.metric("Total Credit Requested", f"{total_credit_requested:,.0f} DM")
    k4.metric("Capital Protected (Rejected)", f"{credit_at_risk:,.0f} DM", delta="High Risk", delta_color="inverse")

    st.markdown("---")

    # ── Row 1: Donut chart + Scatter ───────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Decision Breakdown")
        donut = px.pie(
            pred_df, names='Loan_Decision', hole=0.55,
            color='Loan_Decision', color_discrete_map=RISK_COLORS
        )
        donut.update_traces(textposition='outside', textinfo='percent+label')
        donut.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=380)
        st.plotly_chart(donut, width='stretch')

    with col2:
        st.subheader("Risk Profile Map")
        fig_scatter = px.scatter(
            pred_df, x='Age', y='Credit_Amount', color='Loan_Decision',
            size='Default_Probability', color_discrete_map=RISK_COLORS,
            opacity=0.7, labels={'Credit_Amount': 'Credit Amount (DM)', 'Age': 'Applicant Age'},
            height=420
        )
        fig_scatter.update_layout(margin=dict(t=20, b=0, l=0, r=0))
        st.plotly_chart(fig_scatter, width='stretch')

    st.markdown("---")

    # ── Export ────────────────────────────────────────────────────────────
    high_risk_df = pred_df[pred_df['Loan_Decision'] == 'Rejected']
    csv_data = high_risk_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label=f"⬇️ Export High-Risk Applicants ({len(high_risk_df):,} records)",
        data=csv_data, file_name="high_risk_applicants.csv", mime="text/csv",
        help="Download applicants flagged for manual review."
    )

# ==========================================
# PAGE 2: Model Diagnostics
# ==========================================
elif page == "🔬 Model Diagnostics":
    st.title("🔬 Model Diagnostics")
    st.markdown("*Technical validation of the XGBoost champion model (Test Set Performance).*")
    st.markdown("---")

    # Calculate metrics on the fly using the saved test data
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    rec = recall_score(y_test, y_pred_xgb, pos_label=1)
    prec = precision_score(y_test, y_pred_xgb, pos_label=1)
    f1 = f1_score(y_test, y_pred_xgb, pos_label=1)

    c1, c2, c3 = st.columns(3)
    c1.metric("XGBoost Recall (Bad Loans)", f"{rec:.1%}", help="Percentage of actual bad loans successfully caught.")
    c2.metric("XGBoost Precision", f"{prec:.1%}", help="Accuracy when predicting a bad loan.")
    c3.metric("XGBoost F1-Score", f"{f1:.3f}", help="Harmonic mean of Precision and Recall.")

    st.markdown("---")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Champion Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_xgb)
        cm_fig = px.imshow(
            cm, text_auto=True, color_continuous_scale='Reds',
            labels=dict(x='AI Predicted', y='Actual Outcome'),
            x=['Predicted Good', 'Predicted Bad'], y=['Actual Good', 'Actual Bad'],
            height=400
        )
        cm_fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(cm_fig, width='stretch')

    with colB:
        st.subheader("Feature Importance (XGBoost)")
        # Get feature names from the dataframe used for training
        feature_names = feat_df.drop(['Risk', 'Applicant_ID'], axis=1).columns
        importances = xgb_model.feature_importances_
        
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=False).head(10) # Top 10

        fi_fig = px.bar(
            fi_df, x='Importance', y='Feature', orientation='h',
            text_auto='.3f', color='Importance', color_continuous_scale='Teal',
            height=400
        )
        fi_fig.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
        st.plotly_chart(fi_fig, width='stretch')

# ==========================================
# PAGE 3: Decision Engine
# ==========================================
elif page == "🎯 Decision Engine":
    st.title("🎯 Live Decision Engine")
    st.markdown("*Real-time default probability scored against individual applicants.*")
    st.markdown("---")

    # ── Customer selector ──────────────────────────────────────────────────────
    selected_id = st.selectbox("Select Applicant ID", pred_df['Applicant_ID'].tolist())

    # Pull data
    applicant_raw = pred_df[pred_df['Applicant_ID'] == selected_id].iloc[0]
    
    # Prep features for live prediction
    X_applicant = feat_df[feat_df['Applicant_ID'] == selected_id].drop(['Risk', 'Applicant_ID'], axis=1)
    X_applicant_scaled = scaler.transform(X_applicant)

    # ── Predictions ─────────────────────────────────────────
    xgb_prob = xgb_model.predict_proba(X_applicant_scaled)[0, 1]
    rf_prob = rf_model.predict_proba(X_applicant_scaled)[0, 1]
    lr_prob = lr_model.predict_proba(X_applicant_scaled)[0, 1]

    st.markdown("---")

    col_gauge, col_info = st.columns([1, 1])

    with col_gauge:
        st.subheader("Default Risk Gauge")
        gauge_color = "#E74C3C" if xgb_prob >= 0.35 else "#2ECC71"
        
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=xgb_prob * 100,
            number={'suffix': '%', 'font': {'size': 48}},
            title={'text': "XGBoost (Champion Model)", 'font': {'size': 16}},
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
        st.plotly_chart(gauge_fig, width='stretch')

    with col_info:
        st.subheader("Applicant Profile")
        
        verdict = "🔴 REJECTED (High Risk)" if xgb_prob >= 0.35 else "🟢 APPROVED (Low Risk)"
        st.markdown(f"**System Decision:** {verdict}")
        st.markdown(f"**Actual Historical Outcome:** {'🔴 Defaulted' if applicant_raw['Risk'] == 1 else '🟢 Paid Duly'}")
        
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Age", f"{applicant_raw['Age']} yrs")
        m2.metric("Duration", f"{applicant_raw['Duration_Months']} mo")
        m3.metric("Amount", f"{applicant_raw['Credit_Amount']} DM")

        st.markdown("---")
        # Safely fetch the columns, defaulting to 'Not specified' if the exact column name doesn't exist
        purpose_val = applicant_raw.get('Purpose', applicant_raw.get('purpose', 'Not specified'))
        job_val = applicant_raw.get('Job', applicant_raw.get('job', 'Not specified'))
        
        st.info(f"**Purpose:** {purpose_val}  \n**Job:** {job_val}")

    st.markdown("---")

    st.subheader("Algorithm Consensus")
    compare_fig = px.bar(
        pd.DataFrame({
            'Algorithm': ['XGBoost (Champion)', 'Random Forest', 'Logistic Regression'],
            'Probability': [xgb_prob, rf_prob, lr_prob]
        }),
        x='Algorithm', y='Probability', text_auto='.1%', height=300,
        color='Algorithm', color_discrete_sequence=['#E74C3C', '#BDC3C7', '#BDC3C7']
    )
    compare_fig.add_hline(y=0.35, line_dash='dash', line_color='black', annotation_text='Rejection Threshold (35%)')
    compare_fig.update_layout(yaxis=dict(range=[0, 1], tickformat='.0%'), showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(compare_fig, width='stretch')