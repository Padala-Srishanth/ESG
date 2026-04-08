"""
================================================================================
ESG GREENWASHING DETECTION -- STREAMLIT DASHBOARD
================================================================================
Author  : ML Engineering Scientist (20+ years industry experience)
Project : ESG Greenwashing Detection via NLP & Machine Learning

Fully interactive web dashboard — ALL charts are interactive Plotly charts.
No static images. Every plot is zoomable, hoverable, and filterable.

Run with:
    python -m streamlit run streamlit_dashboard.py
================================================================================
"""

import streamlit as st                       # Streamlit web framework
import pandas as pd                          # Data manipulation
import numpy as np                           # Numerical computing
import plotly.express as px                  # Plotly Express for interactive charts
import plotly.graph_objects as go            # Plotly graph objects for custom charts
from plotly.subplots import make_subplots    # Subplots for multi-panel figures
import os                                    # File system operations
import re                                    # Regular expressions
import json                                  # JSON parsing

# Load .env file if present (for API keys)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_env_path):
    with open(_env_path, 'r') as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _key, _val = _line.split('=', 1)
                os.environ[_key.strip()] = _val.strip()
import xml.etree.ElementTree as ET           # XML parsing for RSS feeds
from datetime import datetime, timedelta     # Date handling
from urllib.parse import quote               # URL encoding

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(                          # Configure page settings
    page_title="ESG Greenwashing Detector",  # Browser tab title
    page_icon="🌍",                          # Browser tab icon
    layout="wide",                           # Use full screen width
    initial_sidebar_state="expanded",        # Sidebar starts open
)

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # Project root
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed") # Processed data dir


# ============================================================================
# DATA LOADING (cached for performance)
# ============================================================================

@st.cache_data                               # Cache data so it loads only once
def load_data():
    """Load all required datasets for the dashboard."""

    data = {}                                # Dictionary to store all datasets

    # Risk scores (main dataset for the dashboard)
    risk_path = os.path.join(PROCESSED_DIR, "greenwashing_risk_scores_expanded.csv")
    if not os.path.exists(risk_path):
        risk_path = os.path.join(PROCESSED_DIR, "greenwashing_risk_scores.csv")
    if os.path.exists(risk_path):
        data['risk_scores'] = pd.read_csv(risk_path, index_col=0)
    else:
        data['risk_scores'] = pd.DataFrame()

    # Model metrics
    metrics_path = os.path.join(PROCESSED_DIR, "model_metrics_expanded.csv")
    if not os.path.exists(metrics_path):
        metrics_path = os.path.join(PROCESSED_DIR, "model_metrics.csv")
    if os.path.exists(metrics_path):
        data['model_metrics'] = pd.read_csv(metrics_path)
    else:
        data['model_metrics'] = pd.DataFrame()

    # Feature importance (best model)
    fi_path = os.path.join(PROCESSED_DIR, "feature_importance_expanded.csv")
    if not os.path.exists(fi_path):
        fi_path = os.path.join(PROCESSED_DIR, "feature_importance_gradient_boosting.csv")
    if os.path.exists(fi_path):
        data['feature_importance'] = pd.read_csv(fi_path)
    else:
        fi_path2 = os.path.join(PROCESSED_DIR, "feature_importance_random_forest.csv")
        if os.path.exists(fi_path2):
            data['feature_importance'] = pd.read_csv(fi_path2)
        else:
            data['feature_importance'] = pd.DataFrame()

    # Feature matrix (for detailed analysis)
    fm_path = os.path.join(PROCESSED_DIR, "feature_matrix_expanded.csv")
    if not os.path.exists(fm_path):
        fm_path = os.path.join(PROCESSED_DIR, "feature_matrix.csv")
    if os.path.exists(fm_path):
        data['feature_matrix'] = pd.read_csv(fm_path)
    else:
        data['feature_matrix'] = pd.DataFrame()

    # Predictions (for confusion matrix data)
    pred_path = os.path.join(PROCESSED_DIR, "predictions_expanded.csv")
    if not os.path.exists(pred_path):
        pred_path = os.path.join(PROCESSED_DIR, "predictions.csv")
    if os.path.exists(pred_path):
        data['predictions'] = pd.read_csv(pred_path)
    else:
        data['predictions'] = pd.DataFrame()

    # SHAP explanations (legacy text file - kept for backward compatibility)
    shap_path = os.path.join(PROCESSED_DIR, "shap_explanations.txt")
    if os.path.exists(shap_path):
        with open(shap_path, 'r', encoding='utf-8') as f:
            data['shap_explanations'] = f.read()
    else:
        data['shap_explanations'] = "SHAP explanations not yet generated. Run regenerate_shap.py."

    # NEW: REAL SHAP feature importance (regenerated on current 205-feature set)
    shap_fi_path = os.path.join(PROCESSED_DIR, "shap_feature_importance.csv")
    if os.path.exists(shap_fi_path):
        data['shap_feature_importance'] = pd.read_csv(shap_fi_path)
    else:
        data['shap_feature_importance'] = pd.DataFrame()

    # NEW: Per-company SHAP values for ALL 480 companies
    shap_all_path = os.path.join(PROCESSED_DIR, "shap_values_all.csv")
    if os.path.exists(shap_all_path):
        data['shap_values_all'] = pd.read_csv(shap_all_path)
    else:
        data['shap_values_all'] = pd.DataFrame()

    # NEW: Hybrid XGBoost-CatBoost model outputs
    hybrid_pred_path = os.path.join(PROCESSED_DIR, "hybrid_predictions.csv")
    if os.path.exists(hybrid_pred_path):
        data['hybrid_predictions'] = pd.read_csv(hybrid_pred_path)
    else:
        data['hybrid_predictions'] = pd.DataFrame()

    hybrid_metrics_path = os.path.join(PROCESSED_DIR, "hybrid_model_metrics.json")
    if os.path.exists(hybrid_metrics_path):
        import json as _json
        with open(hybrid_metrics_path, 'r') as _f:
            data['hybrid_metrics'] = _json.load(_f)
    else:
        data['hybrid_metrics'] = {}

    hybrid_fi_path = os.path.join(PROCESSED_DIR, "hybrid_feature_importance.csv")
    if os.path.exists(hybrid_fi_path):
        data['hybrid_feature_importance'] = pd.read_csv(hybrid_fi_path)
    else:
        data['hybrid_feature_importance'] = pd.DataFrame()

    hybrid_shap_path = os.path.join(PROCESSED_DIR, "hybrid_shap_feature_importance.csv")
    if os.path.exists(hybrid_shap_path):
        data['hybrid_shap_importance'] = pd.read_csv(hybrid_shap_path)
    else:
        data['hybrid_shap_importance'] = pd.DataFrame()

    hybrid_shap_all_path = os.path.join(PROCESSED_DIR, "hybrid_shap_values_all.csv")
    if os.path.exists(hybrid_shap_all_path):
        data['hybrid_shap_values_all'] = pd.read_csv(hybrid_shap_all_path)
    else:
        data['hybrid_shap_values_all'] = pd.DataFrame()

    return data


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar(data):
    """Render the sidebar with filters and navigation."""

    st.sidebar.title("ESG Greenwashing Detector")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.radio(
        "Navigate to:",
        [
            "Risk Score Dashboard",
            "Model Performance",
            "Feature Importance",
            "Company Deep Dive",
            "Company Search & Analysis",
            "Real-Time Intelligence",
            "ESG Report Analyzer (AI)",
            "Report Generator",
            "Advanced Explainability",
            "Company Comparison",
            "Time-Series Risk Tracking",
            "SHAP Explanations",
            "Hybrid XGBoost-CatBoost Model",
        ]
    )

    st.sidebar.markdown("---")

    # Filters
    filters = {}

    if not data['risk_scores'].empty:
        sectors = ['All'] + sorted(data['risk_scores']['sector'].dropna().unique().tolist())
        filters['sector'] = st.sidebar.selectbox("Filter by Sector", sectors)

        if 'risk_tier' in data['risk_scores'].columns:
            tiers = ['All'] + sorted(data['risk_scores']['risk_tier'].dropna().unique().tolist())
            filters['tier'] = st.sidebar.selectbox("Filter by Risk Tier", tiers)

        filters['score_range'] = st.sidebar.slider(
            "Risk Score Range", 0.0, 100.0, (0.0, 100.0), step=1.0
        )

    st.sidebar.markdown("---")
    st.sidebar.caption("Launch: `python -m streamlit run streamlit_dashboard.py`")

    return page, filters


# ============================================================================
# PAGE 1: RISK SCORE DASHBOARD
# ============================================================================

def page_risk_dashboard(data, filters):
    """Main risk score dashboard — all interactive Plotly charts."""

    st.title("Greenwashing Risk Score Dashboard")
    st.markdown("Comprehensive risk assessment of **835 companies** across S&P 500, NIFTY 50, and Public ESG Ratings")

    df = data['risk_scores'].copy()

    if df.empty:
        st.warning("Risk scores not available. Run `python risk_scoring.py` first.")
        return

    # Apply filters
    if filters.get('sector') and filters['sector'] != 'All':
        df = df[df['sector'] == filters['sector']]
    if filters.get('tier') and filters['tier'] != 'All':
        df = df[df['risk_tier'] == filters['tier']]
    if filters.get('score_range'):
        df = df[(df['risk_score'] >= filters['score_range'][0]) &
                (df['risk_score'] <= filters['score_range'][1])]

    # --- KPI Cards ---
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Companies", len(df))
    with col2:
        st.metric("Avg Risk Score", f"{df['risk_score'].mean():.1f}")
    with col3:
        high_risk = len(df[df['risk_score'] >= 60])
        st.metric("High Risk", high_risk)
    with col4:
        st.metric("Max Score", f"{df['risk_score'].max():.1f}")
    with col5:
        st.metric("Median Score", f"{df['risk_score'].median():.1f}")

    st.markdown("---")

    # --- Row 1: Distribution + Tier Pie ---
    col1, col2 = st.columns([2, 1])

    with col1:
        # Interactive histogram with hover info
        fig = px.histogram(
            df, x='risk_score', nbins=25,
            title='Risk Score Distribution',
            labels={'risk_score': 'Greenwashing Risk Score', 'count': 'Number of Companies'},
            color_discrete_sequence=['#e74c3c'],
            hover_data=['company_name'] if 'company_name' in df.columns else None,
        )
        fig.update_layout(showlegend=False, bargap=0.05)
        fig.update_traces(hovertemplate='Score Range: %{x}<br>Count: %{y}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Interactive pie chart
        if 'risk_tier' in df.columns:
            tier_counts = df['risk_tier'].value_counts().reset_index()
            tier_counts.columns = ['Risk Tier', 'Count']
            color_map = {
                'Very Low Risk': '#2ecc71', 'Low Risk': '#82e0aa',
                'Moderate Risk': '#f4d03f', 'High Risk': '#e67e22',
                'Very High Risk': '#e74c3c',
            }
            fig = px.pie(
                tier_counts, values='Count', names='Risk Tier',
                title='Risk Tier Breakdown',
                color='Risk Tier', color_discrete_map=color_map,
                hole=0.4,
            )
            fig.update_traces(textposition='inside', textinfo='percent+label',
                              hovertemplate='%{label}<br>Count: %{value}<br>%{percent}<extra></extra>')
            st.plotly_chart(fig, use_container_width=True)

    # --- Row 2: Sector comparison (interactive bar) ---
    if 'sector' in df.columns:
        sector_stats = df.groupby('sector')['risk_score'].agg(['mean', 'max', 'min', 'count']).reset_index()
        sector_stats.columns = ['Sector', 'Avg Score', 'Max Score', 'Min Score', 'Companies']
        sector_stats = sector_stats.sort_values('Avg Score', ascending=False)

        fig = px.bar(
            sector_stats, x='Sector', y='Avg Score',
            title='Average Risk Score by Sector (hover for details)',
            color='Avg Score', color_continuous_scale='RdYlGn_r',
            hover_data=['Max Score', 'Min Score', 'Companies'],
        )
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>Avg: %{y:.1f}<br>Max: %{customdata[0]:.1f}'
                          '<br>Min: %{customdata[1]:.1f}<br>Companies: %{customdata[2]}<extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Row 3: Scatter — ESG Risk vs Controversy (bubble chart) ---
    if 'total_esg_risk_score' in df.columns and 'controversy_score' in df.columns:
        fig = px.scatter(
            df, x='total_esg_risk_score', y='controversy_score',
            size='risk_score', color='risk_score',
            hover_name='company_name',
            hover_data=['sector', 'risk_tier', 'risk_score'],
            title='ESG Risk vs Controversy (bubble size = risk score)',
            labels={'total_esg_risk_score': 'Total ESG Risk Score',
                    'controversy_score': 'Controversy Score'},
            color_continuous_scale='RdYlGn_r',
            size_max=25,
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # --- Row 4: Interactive Data Table ---
    st.subheader("Company Risk Scores (ranked, sortable)")
    display_cols = ['company_name', 'sector', 'risk_score', 'risk_tier']
    optional = ['total_esg_risk_score', 'controversy_score', 'gw_proxy_score']
    for c in optional:
        if c in df.columns:
            display_cols.append(c)
    display_cols = [c for c in display_cols if c in df.columns]

    st.dataframe(
        df[display_cols].reset_index(drop=True),
        use_container_width=True, height=400,
    )


# ============================================================================
# PAGE 2: MODEL PERFORMANCE (all interactive)
# ============================================================================

def page_model_performance(data):
    """Model performance comparison — fully interactive Plotly charts."""

    st.title("Model Performance Comparison")

    metrics = data['model_metrics']
    if metrics.empty:
        st.warning("Model metrics not available. Run `python model_training.py` first.")
        return

    # KPI cards for best model
    best = metrics.iloc[0]
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Best Model", best['model'])
    with col2:
        st.metric("F1 Score", f"{best['f1_score']:.4f}")
    with col3:
        st.metric("Accuracy", f"{best['accuracy']:.4f}")
    with col4:
        st.metric("ROC-AUC", f"{best['roc_auc']:.4f}")
    with col5:
        st.metric("Precision", f"{best['precision']:.4f}")

    st.markdown("---")

    # --- Interactive grouped bar chart ---
    metric_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    available_metrics = [c for c in metric_cols if c in metrics.columns]

    melted = metrics.melt(id_vars=['model'], value_vars=available_metrics,
                          var_name='Metric', value_name='Score')
    fig = px.bar(
        melted, x='model', y='Score', color='Metric',
        barmode='group', title='Model Performance Comparison (click legend to toggle metrics)',
        color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'],
        text='Score',
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(yaxis_range=[0, 1.15], height=500)
    st.plotly_chart(fig, use_container_width=True)

    # --- Interactive radar chart for model comparison ---
    st.subheader("Model Radar Comparison")

    fig = go.Figure()
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    for idx, (_, row) in enumerate(metrics.iterrows()):
        values = [row.get(m, 0) for m in available_metrics]
        values.append(values[0])  # Close the radar polygon
        categories = available_metrics + [available_metrics[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=row['model'],
            line=dict(color=colors[idx % len(colors)]),
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title='Radar: Model Strengths Across All Metrics',
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Interactive confusion matrix heatmaps ---
    st.subheader("Confusion Matrix Analysis")

    # Reconstruct confusion matrix data from predictions
    pred_df = data.get('predictions', pd.DataFrame())
    if not pred_df.empty and 'gw_label_binary' in pred_df.columns and 'gw_proxy_score' in pred_df.columns:
        # Show proxy score distribution as interactive histogram
        fig = px.histogram(
            pred_df, x='gw_proxy_score', nbins=6,
            title='Greenwashing Proxy Score Distribution (0-5 indicators)',
            labels={'gw_proxy_score': 'Proxy Score (0-5)', 'count': 'Companies'},
            color_discrete_sequence=['#3498db'],
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

    # --- Training time comparison ---
    if 'training_time_sec' in metrics.columns:
        fig = px.bar(
            metrics.sort_values('training_time_sec', ascending=True),
            x='training_time_sec', y='model', orientation='h',
            title='Training Time Comparison (seconds)',
            labels={'training_time_sec': 'Time (seconds)', 'model': 'Model'},
            color='training_time_sec', color_continuous_scale='Blues',
            text='training_time_sec',
        )
        fig.update_traces(texttemplate='%{text:.1f}s', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    # --- Full metrics table ---
    st.subheader("Detailed Metrics Table")
    st.dataframe(metrics, use_container_width=True)


# ============================================================================
# PAGE 3: FEATURE IMPORTANCE (all interactive)
# ============================================================================

def page_feature_importance(data):
    """Feature importance — fully interactive Plotly charts."""

    st.title("Feature Importance Analysis")

    fi = data['feature_importance']
    if fi.empty:
        st.warning("Feature importance not available. Run model training first.")
        return

    # Top N selector
    top_n = st.slider("Number of features to show", 5, 50, 20)

    top_fi = fi.head(top_n)

    # --- Interactive horizontal bar chart ---
    fig = px.bar(
        top_fi, x='importance', y='feature', orientation='h',
        title=f'Top {top_n} Most Important Features (hover for details)',
        labels={'importance': 'Feature Importance', 'feature': 'Feature'},
        color='importance', color_continuous_scale='Reds',
        text='importance',
    )
    fig.update_traces(texttemplate='%{text:.5f}', textposition='outside')
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=max(500, top_n * 28),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Feature importance treemap ---
    st.subheader("Feature Importance Treemap")
    top30 = fi.head(30).copy()
    top30['importance_pct'] = (top30['importance'] / top30['importance'].sum() * 100).round(2)

    # Categorize features by type
    def categorize_feature(name):
        if any(k in name for k in ['controversy', 'esg_risk', 'pillar', 'env_', 'social_', 'gov_']):
            return 'Numerical ESG'
        elif any(k in name for k in ['greenwashing', 'vague', 'hedge', 'linguistic', 'sentiment', 'word']):
            return 'NLP Linguistic'
        elif any(k in name for k in ['sector', 'industry', 'bin', 'tier', 'flag', 'segment']):
            return 'Categorical'
        elif 'scaled' in name:
            return 'Scaled'
        else:
            return 'Other'

    top30['category'] = top30['feature'].apply(categorize_feature)

    fig = px.treemap(
        top30, path=['category', 'feature'], values='importance_pct',
        title='Feature Importance Treemap (grouped by category)',
        color='importance_pct', color_continuous_scale='Reds',
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # --- Cumulative importance curve ---
    st.subheader("Cumulative Feature Importance")
    fi_sorted = fi.copy()
    fi_sorted['cumulative'] = fi_sorted['importance'].cumsum()
    fi_sorted['cumulative_pct'] = (fi_sorted['cumulative'] / fi_sorted['importance'].sum() * 100)
    fi_sorted['rank'] = range(1, len(fi_sorted) + 1)

    fig = px.line(
        fi_sorted, x='rank', y='cumulative_pct',
        title='Cumulative Feature Importance (how many features explain N% of prediction)',
        labels={'rank': 'Number of Features', 'cumulative_pct': 'Cumulative Importance (%)'},
        markers=True,
    )
    fig.add_hline(y=90, line_dash="dash", line_color="red",
                  annotation_text="90% threshold", annotation_position="top left")
    fig.add_hline(y=95, line_dash="dash", line_color="orange",
                  annotation_text="95% threshold", annotation_position="bottom right")
    st.plotly_chart(fig, use_container_width=True)

    # n features for 90%
    n_90 = fi_sorted[fi_sorted['cumulative_pct'] >= 90].iloc[0]['rank'] if len(fi_sorted[fi_sorted['cumulative_pct'] >= 90]) > 0 else len(fi_sorted)
    st.info(f"Top **{int(n_90)} features** explain **90%** of the model's predictive power "
            f"(out of {len(fi_sorted)} total features).")

    # --- Feature importance table ---
    st.subheader("Full Feature Importance Table")
    st.dataframe(fi, use_container_width=True, height=400)


# ============================================================================
# PAGE 4: COMPANY DEEP DIVE (all interactive)
# ============================================================================

def page_company_deep_dive(data):
    """Company-level deep dive — fully interactive charts."""

    st.title("Company Deep Dive")

    df = data['risk_scores']
    fm = data['feature_matrix']

    if df.empty:
        st.warning("Risk scores not available. Run risk_scoring.py first.")
        return

    # Company selector
    company_names = sorted(df['company_name'].dropna().unique().tolist())
    selected = st.selectbox("Select a company to investigate", company_names)

    if selected:
        company = df[df['company_name'] == selected].iloc[0]

        # Company header KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Risk Score", f"{company['risk_score']:.1f}/100")
        with col2:
            st.metric("Risk Tier", str(company.get('risk_tier', 'N/A')))
        with col3:
            st.metric("ESG Risk Score", f"{company.get('total_esg_risk_score', 'N/A')}")
        with col4:
            st.metric("Controversy", f"{company.get('controversy_score', 'N/A')}")

        st.markdown("---")

        # --- Interactive component breakdown (bar chart) ---
        st.subheader("Risk Score Component Breakdown")
        components = {}
        comp_cols = {
            'comp_proxy': 'Proxy Score', 'comp_linguistic': 'Linguistic',
            'comp_divergence': 'ESG-Controversy Gap',
            'comp_credibility_inv': 'Low Credibility',
            'comp_controversy_ratio': 'Controversy Ratio',
        }
        for col, nice_name in comp_cols.items():
            if col in company.index:
                val = company[col]
                if pd.notna(val):
                    components[nice_name] = float(val)

        if components:
            comp_df = pd.DataFrame({
                'Component': list(components.keys()),
                'Score': list(components.values()),
                'Weight': ['40%', '15%', '15%', '15%', '15%'][:len(components)],
            })
            fig = px.bar(
                comp_df, x='Component', y='Score',
                title=f'Risk Score Breakdown -- {selected}',
                color='Score', color_continuous_scale='RdYlGn_r',
                text='Score', hover_data=['Weight'],
            )
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig.update_layout(yaxis_range=[0, 110])
            st.plotly_chart(fig, use_container_width=True)

        # --- Interactive gauge chart for overall risk ---
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=company['risk_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Greenwashing Risk: {selected}", 'font': {'size': 18}},
            delta={'reference': df['risk_score'].mean(), 'increasing': {'color': "red"},
                   'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 20], 'color': '#2ecc71'},
                    {'range': [20, 40], 'color': '#82e0aa'},
                    {'range': [40, 60], 'color': '#f4d03f'},
                    {'range': [60, 80], 'color': '#e67e22'},
                    {'range': [80, 100], 'color': '#e74c3c'},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': df['risk_score'].mean(),
                },
            }
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Black line = population average ({df['risk_score'].mean():.1f}). "
                   f"Delta shows difference from average.")

        # --- Key features radar chart ---
        if not fm.empty and selected in fm['company_name'].values:
            st.subheader("Key Feature Profile")
            company_fm = fm[fm['company_name'] == selected].iloc[0]

            feature_pairs = [
                ('total_esg_risk_score', 'ESG Risk'),
                ('env_risk_score', 'Environmental'),
                ('social_risk_score', 'Social'),
                ('gov_risk_score', 'Governance'),
                ('controversy_score', 'Controversy'),
            ]

            available_pairs = [(col, label) for col, label in feature_pairs if col in fm.columns]

            if available_pairs:
                # Normalize values to 0-1 for radar
                categories = [label for _, label in available_pairs]
                company_vals = []
                population_avg = []

                for col, _ in available_pairs:
                    col_min = fm[col].min()
                    col_max = fm[col].max()
                    norm = (company_fm[col] - col_min) / (col_max - col_min + 1e-8)
                    avg_norm = (fm[col].mean() - col_min) / (col_max - col_min + 1e-8)
                    company_vals.append(round(norm, 3))
                    population_avg.append(round(avg_norm, 3))

                # Close the polygon
                categories_closed = categories + [categories[0]]
                company_vals_closed = company_vals + [company_vals[0]]
                population_avg_closed = population_avg + [population_avg[0]]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=company_vals_closed, theta=categories_closed,
                    fill='toself', name=selected,
                    line=dict(color='#e74c3c'), opacity=0.7,
                ))
                fig.add_trace(go.Scatterpolar(
                    r=population_avg_closed, theta=categories_closed,
                    fill='toself', name='Population Average',
                    line=dict(color='#3498db', dash='dash'), opacity=0.4,
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title=f'ESG Profile: {selected} vs Population Average',
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

            # --- Feature detail table ---
            st.subheader("Detailed Feature Values")
            key_features = {
                'Total ESG Risk Score': company_fm.get('total_esg_risk_score'),
                'Environmental Risk': company_fm.get('env_risk_score'),
                'Social Risk': company_fm.get('social_risk_score'),
                'Governance Risk': company_fm.get('gov_risk_score'),
                'Controversy Score': company_fm.get('controversy_score'),
                'Pillar Imbalance': company_fm.get('pillar_imbalance_score'),
                'ESG-Controversy Divergence': company_fm.get('esg_controversy_divergence'),
                'GW Linguistic Score': company_fm.get('greenwashing_signal_score'),
                'Controversy Risk Ratio': company_fm.get('controversy_risk_ratio'),
                'Combined Anomaly Score': company_fm.get('combined_anomaly_score'),
                'Aggregate ESG NLP Score': company_fm.get('aggregate_esg_nlp_score'),
                'Regulatory Readiness': company_fm.get('regulatory_readiness_score'),
                'Policy-ESG Gap': company_fm.get('policy_esg_gap'),
                'Narrative Credibility': company_fm.get('narrative_credibility_index'),
                'Commitment Credibility': company_fm.get('commitment_credibility_score'),
                'Multi-Signal GW Score': company_fm.get('multi_signal_greenwashing_score'),
            }
            kf_df = pd.DataFrame({
                'Feature': list(key_features.keys()),
                'Value': [f"{v:.4f}" if isinstance(v, (int, float)) and not pd.isna(v) else 'N/A'
                          for v in key_features.values()],
            })
            st.table(kf_df)

        # --- Sector peer comparison ---
        if 'sector' in df.columns:
            sector = company.get('sector')
            if sector:
                st.subheader(f"Peer Comparison: {sector} Sector")
                peers = df[df['sector'] == sector].copy()
                peers['is_selected'] = peers['company_name'] == selected

                fig = px.bar(
                    peers.sort_values('risk_score', ascending=False),
                    x='company_name', y='risk_score',
                    color='is_selected',
                    color_discrete_map={True: '#e74c3c', False: '#bdc3c7'},
                    title=f'Risk Scores in {sector} Sector (red = selected company)',
                    labels={'risk_score': 'Risk Score', 'company_name': 'Company'},
                    hover_data=['risk_tier'],
                )
                fig.update_layout(showlegend=False, xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE 5: COMPANY SEARCH & ANALYSIS
# ============================================================================

def page_company_search(data):
    """Search any company by name and get a full greenwashing analysis report."""

    st.title("Company Search & Analysis")
    st.markdown("Select a company to get a **complete greenwashing risk analysis report**.")

    df = data['risk_scores']
    fm = data['feature_matrix']
    pred = data['predictions']

    if df.empty or fm.empty:
        st.warning("Data not available. Run `python model_pipeline.py` first.")
        return

    # --- Searchable dropdown with all companies ---
    company_names = sorted(df['company_name'].dropna().unique().tolist())
    selected = st.selectbox(
        "Search & Select Company",
        options=company_names,
        index=None,
        placeholder="Start typing to search (e.g., Apple, Tesla, Adani)...",
        key='company_search_select',
    )

    if not selected:
        st.info("Select a company from the dropdown above. Start typing to filter the list.")
        # Show top 10 highest risk companies as suggestions
        st.subheader("Top 10 Highest Risk Companies")
        top10 = df.nlargest(10, 'risk_score')[['company_name', 'sector', 'risk_score', 'risk_tier']]
        st.dataframe(top10.reset_index(drop=True), use_container_width=True)
        return

    # ===================== FULL ANALYSIS REPORT =====================
    company_risk = df[df['company_name'] == selected].iloc[0]
    company_fm_row = fm[fm['company_name'] == selected]

    if company_fm_row.empty:
        st.error(f"Feature data not found for {selected}.")
        return

    company_fm = company_fm_row.iloc[0]

    st.markdown("---")
    st.header(f"Analysis Report: {selected}")

    # --- Section 1: Overview KPIs ---
    st.subheader("1. Risk Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        risk_score = company_risk['risk_score']
        st.metric("Risk Score", f"{risk_score:.1f}/100",
                  delta=f"{risk_score - df['risk_score'].mean():.1f} vs avg",
                  delta_color="inverse")
    with col2:
        st.metric("Risk Tier", str(company_risk.get('risk_tier', 'N/A')))
    with col3:
        st.metric("ESG Risk Score", f"{company_risk.get('total_esg_risk_score', 0):.1f}")
    with col4:
        st.metric("Controversy Score", f"{company_risk.get('controversy_score', 0):.1f}")
    with col5:
        rank_val = company_risk.get('rank', 'N/A')
        st.metric("Risk Rank", f"#{int(rank_val)}/{len(df)}" if rank_val != 'N/A' else 'N/A')

    # --- Verdict banner ---
    if risk_score >= 60:
        st.error(f"HIGH GREENWASHING RISK -- {selected} scores {risk_score:.1f}/100. "
                 f"Multiple indicators suggest potential greenwashing behavior.")
    elif risk_score >= 40:
        st.warning(f"MODERATE GREENWASHING RISK -- {selected} scores {risk_score:.1f}/100. "
                   f"Some indicators warrant further investigation.")
    else:
        st.success(f"LOW GREENWASHING RISK -- {selected} scores {risk_score:.1f}/100. "
                   f"ESG claims appear largely consistent with actual performance.")

    # ===================== REASON ENGINE =====================
    st.markdown("---")
    st.subheader("Why This Rating?")
    st.markdown("""
    <style>
    .reason-box {
        background: #1a1a2e; border-radius: 12px; padding: 20px 25px;
        margin: 10px 0; border: 1px solid #333;
    }
    .reason-title {
        font-family: monospace; font-size: 18px; font-weight: bold; margin-bottom: 15px;
    }
    .reason-title-high { color: #ff4444; }
    .reason-title-moderate { color: #ffaa00; }
    .reason-title-low { color: #00cc66; }
    .reason-item {
        padding: 8px 0; border-bottom: 1px solid #2a2a3e; font-size: 14px; color: #e0e0e0;
    }
    .reason-item:last-child { border-bottom: none; }
    .reason-icon { margin-right: 10px; }
    .reason-detail { color: #999; font-size: 12px; margin-left: 28px; }
    .reason-tag {
        display: inline-block; padding: 2px 8px; border-radius: 10px;
        font-size: 11px; font-weight: bold; margin-left: 8px;
    }
    .tag-critical { background: #dc2626; color: white; }
    .tag-warning { background: #d97706; color: white; }
    .tag-good { background: #16a34a; color: white; }
    .tag-info { background: #2563eb; color: white; }
    </style>
    """, unsafe_allow_html=True)

    # --- Build reasons from actual data ---
    reasons = []

    # 1. ESG-Controversy Divergence
    divergence = float(company_fm.get('esg_controversy_divergence', 0)) if not pd.isna(company_fm.get('esg_controversy_divergence', 0)) else 0
    div_pct = float((fm[fm['esg_controversy_divergence'] <= divergence].shape[0] / len(fm)) * 100) if 'esg_controversy_divergence' in fm.columns else 50
    esg_val = float(company_risk.get('total_esg_risk_score', 0))
    controversy_val = float(company_risk.get('controversy_score', 0))
    if divergence > fm['esg_controversy_divergence'].quantile(0.75) if 'esg_controversy_divergence' in fm.columns else 0:
        reasons.append({
            'icon': '🚨', 'severity': 'critical',
            'text': f'ESG risk is low ({esg_val:.1f}) but controversy is high ({controversy_val:.1f}) -- classic greenwashing divergence pattern',
            'detail': f'ESG-Controversy Divergence = {divergence:.3f} (higher than {div_pct:.0f}% of companies). The company claims low ESG risk but has disproportionately high controversy.',
            'tag': 'CRITICAL',
        })
    elif divergence > fm['esg_controversy_divergence'].median() if 'esg_controversy_divergence' in fm.columns else 0:
        reasons.append({
            'icon': '⚠️', 'severity': 'warning',
            'text': f'Moderate gap between ESG claims ({esg_val:.1f}) and controversy level ({controversy_val:.1f})',
            'detail': f'Divergence = {divergence:.3f} (above median). Some inconsistency between stated ESG performance and actual controversies.',
            'tag': 'WARNING',
        })
    else:
        reasons.append({
            'icon': '✅', 'severity': 'good',
            'text': f'ESG risk ({esg_val:.1f}) and controversy ({controversy_val:.1f}) are consistent',
            'detail': f'Divergence = {divergence:.3f} (below median). No significant gap between claims and reality.',
            'tag': 'GOOD',
        })

    # 2. Controversy-Risk Ratio
    crr = float(company_fm.get('controversy_risk_ratio', 0)) if not pd.isna(company_fm.get('controversy_risk_ratio', 0)) else 0
    crr_pct = float((fm[fm['controversy_risk_ratio'] <= crr].shape[0] / len(fm)) * 100) if 'controversy_risk_ratio' in fm.columns else 50
    if crr > fm['controversy_risk_ratio'].quantile(0.75) if 'controversy_risk_ratio' in fm.columns else 0:
        reasons.append({
            'icon': '🚨', 'severity': 'critical',
            'text': f'Controversy-Risk Ratio ({crr:.3f}) is in the top {100 - crr_pct:.0f}% -- controversy disproportionately high',
            'detail': f'This is the #1 predictive feature in the model (importance = 0.276). A high ratio means controversy levels are not reflected in the ESG risk score.',
            'tag': 'CRITICAL',
        })
    elif crr > fm['controversy_risk_ratio'].median() if 'controversy_risk_ratio' in fm.columns else 0:
        reasons.append({
            'icon': '⚠️', 'severity': 'warning',
            'text': f'Controversy-Risk Ratio ({crr:.3f}) is above average',
            'detail': f'Percentile: {crr_pct:.0f}th. Controversy is somewhat elevated relative to ESG risk score.',
            'tag': 'WARNING',
        })

    # 3. Linguistic Greenwashing Signal
    gw_signal = float(company_fm.get('greenwashing_signal_score', 0)) if not pd.isna(company_fm.get('greenwashing_signal_score', 0)) else 0
    vague = int(company_fm.get('vague_language_count', 0)) if not pd.isna(company_fm.get('vague_language_count', 0)) else 0
    concrete = int(company_fm.get('concrete_evidence_count', 0)) if not pd.isna(company_fm.get('concrete_evidence_count', 0)) else 0
    hedge = int(company_fm.get('hedge_language_count', 0)) if not pd.isna(company_fm.get('hedge_language_count', 0)) else 0
    superlative = int(company_fm.get('superlative_count', 0)) if not pd.isna(company_fm.get('superlative_count', 0)) else 0
    future_lang = int(company_fm.get('future_language_count', 0)) if not pd.isna(company_fm.get('future_language_count', 0)) else 0

    if gw_signal > fm['greenwashing_signal_score'].quantile(0.75) if 'greenwashing_signal_score' in fm.columns else 0.5:
        reasons.append({
            'icon': '🚨', 'severity': 'critical',
            'text': f'Corporate text shows strong greenwashing language patterns (GW signal = {gw_signal:.3f})',
            'detail': f'Vague terms: {vague} | Hedge words: {hedge} | Superlatives: {superlative} | Future promises: {future_lang} | Concrete evidence: {concrete}. '
                      + (f'Vague language ({vague}) significantly outweighs concrete evidence ({concrete}).' if vague > concrete else ''),
            'tag': 'CRITICAL',
        })
    elif vague > concrete and vague > 2:
        reasons.append({
            'icon': '⚠️', 'severity': 'warning',
            'text': f'More vague promises ({vague}) than concrete evidence ({concrete}) in corporate text',
            'detail': f'GW signal = {gw_signal:.3f}. Hedge words: {hedge}, Superlatives: {superlative}, Future language: {future_lang}.',
            'tag': 'WARNING',
        })
    elif concrete > vague and concrete > 1:
        reasons.append({
            'icon': '✅', 'severity': 'good',
            'text': f'Corporate text is substantive: {concrete} concrete evidence points vs {vague} vague terms',
            'detail': f'GW signal = {gw_signal:.3f} (low). Language appears credible with measurable claims.',
            'tag': 'GOOD',
        })

    # 4. Pillar Imbalance
    imbalance = float(company_fm.get('pillar_imbalance_score', 0)) if not pd.isna(company_fm.get('pillar_imbalance_score', 0)) else 0
    env_r = float(company_fm.get('env_risk_score', 0)) if not pd.isna(company_fm.get('env_risk_score', 0)) else 0
    social_r = float(company_fm.get('social_risk_score', 0)) if not pd.isna(company_fm.get('social_risk_score', 0)) else 0
    gov_r = float(company_fm.get('gov_risk_score', 0)) if not pd.isna(company_fm.get('gov_risk_score', 0)) else 0
    avg_imbalance = float(fm['pillar_imbalance_score'].mean()) if 'pillar_imbalance_score' in fm.columns else 0

    if imbalance > avg_imbalance * 1.5:
        # Find which pillars are inconsistent
        pillars = {'Environmental': env_r, 'Social': social_r, 'Governance': gov_r}
        highest = max(pillars, key=pillars.get)
        lowest = min(pillars, key=pillars.get)
        gap = pillars[highest] - pillars[lowest]
        reasons.append({
            'icon': '⚠️', 'severity': 'warning',
            'text': f'{highest} risk ({pillars[highest]:.1f}) is inconsistent with {lowest} risk ({pillars[lowest]:.1f}) -- gap of {gap:.1f} points',
            'detail': f'Pillar imbalance = {imbalance:.2f} (avg = {avg_imbalance:.2f}). Uneven ESG performance suggests selective reporting -- strong in one area but weak in another.',
            'tag': 'WARNING',
        })
    else:
        reasons.append({
            'icon': '✅', 'severity': 'good',
            'text': f'ESG pillars are balanced (E={env_r:.1f}, S={social_r:.1f}, G={gov_r:.1f})',
            'detail': f'Imbalance = {imbalance:.2f} (avg = {avg_imbalance:.2f}). No sign of selective reporting.',
            'tag': 'GOOD',
        })

    # 5. Risk-Controversy Mismatch
    mismatch = int(company_fm.get('risk_controversy_mismatch', 0)) if not pd.isna(company_fm.get('risk_controversy_mismatch', 0)) else 0
    if mismatch:
        reasons.append({
            'icon': '🚨', 'severity': 'critical',
            'text': f'Risk-Controversy MISMATCH detected -- ESG risk level does not match controversy level',
            'detail': 'The company\'s ESG risk category and controversy category are in different tiers. This is a key greenwashing indicator -- the company may be masking controversy with favorable ESG scores.',
            'tag': 'CRITICAL',
        })

    # 6. Anomaly Detection
    anomaly = float(company_fm.get('combined_anomaly_score', 0)) if not pd.isna(company_fm.get('combined_anomaly_score', 0)) else 0
    anomaly_thresh = float(fm['combined_anomaly_score'].quantile(0.75)) if 'combined_anomaly_score' in fm.columns else 0
    if anomaly > anomaly_thresh:
        reasons.append({
            'icon': '⚠️', 'severity': 'warning',
            'text': f'Statistical anomaly detected (score = {anomaly:.3f}, threshold = {anomaly_thresh:.3f})',
            'detail': 'This company\'s ESG profile is statistically unusual compared to the population. Anomaly detection (IQR + MAD) flagged multiple features as outliers.',
            'tag': 'WARNING',
        })

    # 7. Sentiment Analysis
    sentiment = float(company_fm.get('text_polarity', 0)) if not pd.isna(company_fm.get('text_polarity', 0)) else 0
    sentiment_thresh = float(fm['text_polarity'].quantile(0.85)) if 'text_polarity' in fm.columns else 0.5
    if sentiment > sentiment_thresh:
        reasons.append({
            'icon': '⚠️', 'severity': 'warning',
            'text': f'Overly positive sentiment in corporate text (polarity = {sentiment:.3f}, top 15%)',
            'detail': 'Excessively positive ESG language can indicate puffery -- making claims sound better than reality. Credible reports tend to have balanced or neutral tone.',
            'tag': 'WARNING',
        })

    # 8. Government Policy Alignment
    reg_readiness = float(company_fm.get('regulatory_readiness_score', 0)) if not pd.isna(company_fm.get('regulatory_readiness_score', 0)) else 0
    policy_gap = float(company_fm.get('policy_esg_gap', 0.5)) if not pd.isna(company_fm.get('policy_esg_gap', 0.5)) else 0.5
    reg_breadth = float(company_fm.get('regulatory_breadth_index', 0)) if not pd.isna(company_fm.get('regulatory_breadth_index', 0)) else 0
    if reg_readiness < 0.15 and policy_gap > 0.6:
        reasons.append({
            'icon': '🚨', 'severity': 'critical',
            'text': f'Very low regulatory readiness ({reg_readiness:.3f}) with high policy-ESG gap ({policy_gap:.3f})',
            'detail': f'Company uses ESG language without grounding in recognized frameworks (Paris Agreement, EU Taxonomy, TCFD, SDGs). '
                      f'Regulatory breadth: {reg_breadth:.0%} of frameworks referenced. This is a strong greenwashing indicator.',
            'tag': 'CRITICAL',
        })
    elif reg_readiness > 0.4:
        reasons.append({
            'icon': '✅', 'severity': 'good',
            'text': f'Strong regulatory alignment (readiness = {reg_readiness:.3f}, breadth = {reg_breadth:.0%})',
            'detail': f'Company references recognized ESG frameworks. Policy-ESG gap = {policy_gap:.3f}.',
            'tag': 'GOOD',
        })

    # 9. News Intent & Narrative Credibility
    narrative_cred = float(company_fm.get('narrative_credibility_index', 0.5)) if not pd.isna(company_fm.get('narrative_credibility_index', 0.5)) else 0.5
    promo_dom = float(company_fm.get('promotional_dominance_score', 0)) if not pd.isna(company_fm.get('promotional_dominance_score', 0)) else 0
    news_gw = float(company_fm.get('news_greenwashing_signal', 0)) if not pd.isna(company_fm.get('news_greenwashing_signal', 0)) else 0
    if promo_dom > 0.6 and narrative_cred < 0.3:
        reasons.append({
            'icon': '⚠️', 'severity': 'warning',
            'text': f'Highly promotional narrative (dominance = {promo_dom:.3f}) with low credibility ({narrative_cred:.3f})',
            'detail': f'Text is dominated by marketing language rather than factual data or strategic substance. News GW signal = {news_gw:.3f}.',
            'tag': 'WARNING',
        })
    elif narrative_cred > 0.6:
        reasons.append({
            'icon': '✅', 'severity': 'good',
            'text': f'Credible narrative profile (credibility = {narrative_cred:.3f})',
            'detail': f'Communication is balanced between factual, strategic, and promotional content. Promotional dominance = {promo_dom:.3f}.',
            'tag': 'GOOD',
        })

    # 10. Temporal Commitment Credibility
    commit_cred = float(company_fm.get('commitment_credibility_score', 0.5)) if not pd.isna(company_fm.get('commitment_credibility_score', 0.5)) else 0.5
    temporal_gw = float(company_fm.get('temporal_greenwashing_signal', 0.5)) if not pd.isna(company_fm.get('temporal_greenwashing_signal', 0.5)) else 0.5
    ppr = float(company_fm.get('progress_to_promise_ratio', 0.5)) if not pd.isna(company_fm.get('progress_to_promise_ratio', 0.5)) else 0.5
    if commit_cred < 0.3 and temporal_gw > 0.7:
        reasons.append({
            'icon': '⚠️', 'severity': 'warning',
            'text': f'Low commitment credibility ({commit_cred:.3f}) -- promises outweigh past achievements',
            'detail': f'Temporal GW signal = {temporal_gw:.3f}. Progress-to-promise ratio = {ppr:.3f}. '
                      f'Company makes future commitments without demonstrating past delivery.',
            'tag': 'WARNING',
        })
    elif commit_cred > 0.7:
        reasons.append({
            'icon': '✅', 'severity': 'good',
            'text': f'Strong commitment credibility ({commit_cred:.3f}) -- backed by past achievements',
            'detail': f'Progress-to-promise ratio = {ppr:.3f}. Past delivery supports future commitments.',
            'tag': 'GOOD',
        })

    # 11. Aggregate ESG NLP Score
    agg_score = float(company_fm.get('aggregate_esg_nlp_score', 50)) if not pd.isna(company_fm.get('aggregate_esg_nlp_score', 50)) else 50
    multi_gw = float(company_fm.get('multi_signal_greenwashing_score', 0.5)) if not pd.isna(company_fm.get('multi_signal_greenwashing_score', 0.5)) else 0.5
    if agg_score < 30:
        reasons.append({
            'icon': '🚨', 'severity': 'critical',
            'text': f'Very low Aggregate ESG NLP Score ({agg_score:.1f}/100) -- multiple greenwashing signals detected',
            'detail': f'Multi-signal GW score = {multi_gw:.3f}. Linguistic, policy, temporal, and narrative dimensions all indicate elevated risk.',
            'tag': 'CRITICAL',
        })
    elif agg_score > 70:
        reasons.append({
            'icon': '✅', 'severity': 'good',
            'text': f'Strong Aggregate ESG NLP Score ({agg_score:.1f}/100) -- credible ESG communication',
            'detail': f'Multi-signal GW score = {multi_gw:.3f}. Policy alignment, factual narrative, and temporal credibility are consistent.',
            'tag': 'GOOD',
        })

    # 12. Sector Ranking
    sector = company_risk.get('sector', None)
    if sector:
        sector_peers = df[df['sector'] == sector].sort_values('risk_score', ascending=False)
        sector_rank = list(sector_peers['company_name']).index(selected) + 1 if selected in sector_peers['company_name'].values else 0
        sector_total = len(sector_peers)
        sector_avg = float(sector_peers['risk_score'].mean())

        if sector_rank <= 3 and sector_total >= 5:
            reasons.append({
                'icon': '🚨', 'severity': 'critical',
                'text': f'Ranked #{sector_rank} highest risk out of {sector_total} companies in {sector} sector',
                'detail': f'Company risk = {risk_score:.1f}, Sector avg = {sector_avg:.1f} (delta = {risk_score - sector_avg:+.1f}). Among the worst performers in its peer group.',
                'tag': 'CRITICAL',
            })
        elif risk_score > sector_avg + 10:
            reasons.append({
                'icon': '⚠️', 'severity': 'warning',
                'text': f'Risk score ({risk_score:.1f}) is {risk_score - sector_avg:.1f} points above {sector} sector average ({sector_avg:.1f})',
                'detail': f'Ranked #{sector_rank}/{sector_total} in sector. Above-average risk compared to industry peers.',
                'tag': 'WARNING',
            })
        elif risk_score < sector_avg - 10:
            reasons.append({
                'icon': '✅', 'severity': 'good',
                'text': f'Risk score ({risk_score:.1f}) is {sector_avg - risk_score:.1f} points below {sector} sector average ({sector_avg:.1f})',
                'detail': f'Ranked #{sector_rank}/{sector_total} in sector. Better than most peers.',
                'tag': 'GOOD',
            })

    # 9. Proxy Score
    if not pred.empty:
        pred_row = pred[pred['company_name'] == selected]
        if not pred_row.empty:
            proxy = int(pred_row.iloc[0].get('gw_proxy_score', 0))
            if proxy >= 3:
                reasons.append({
                    'icon': '🚨', 'severity': 'critical',
                    'text': f'Triggered {proxy} out of 5 greenwashing indicators in the ML model',
                    'detail': 'Indicators: ESG-Controversy Divergence, GW Linguistic Score, Risk-Controversy Mismatch, Controversy-Risk Ratio, Combined Anomaly. Each triggers when above 75th percentile.',
                    'tag': 'CRITICAL',
                })
            elif proxy == 2:
                reasons.append({
                    'icon': '⚠️', 'severity': 'warning',
                    'text': f'Triggered {proxy} out of 5 greenwashing indicators (threshold for flagging)',
                    'detail': 'The model flags companies with 2+ indicators. This company is at the boundary.',
                    'tag': 'WARNING',
                })

    # --- Render the Reason Engine ---
    critical_count = sum(1 for r in reasons if r['severity'] == 'critical')
    warning_count = sum(1 for r in reasons if r['severity'] == 'warning')
    good_count = sum(1 for r in reasons if r['severity'] == 'good')

    if risk_score >= 60:
        title_class = 'reason-title-high'
        title_icon = '🚨'
        title_text = f'{selected} -- HIGH RISK'
    elif risk_score >= 40:
        title_class = 'reason-title-moderate'
        title_icon = '⚠️'
        title_text = f'{selected} -- MODERATE RISK'
    else:
        title_class = 'reason-title-low'
        title_icon = '✅'
        title_text = f'{selected} -- LOW RISK'

    # Build HTML
    reasons_html = f'<div class="reason-box">'
    reasons_html += f'<div class="reason-title {title_class}">{title_icon} {title_text}</div>'
    reasons_html += f'<div style="color:#888;font-size:12px;margin-bottom:12px;">{critical_count} critical | {warning_count} warnings | {good_count} positive signals</div>'

    # Sort: critical first, then warning, then good
    severity_order = {'critical': 0, 'warning': 1, 'good': 2}
    reasons.sort(key=lambda x: severity_order.get(x['severity'], 3))

    for r in reasons:
        tag_class = 'tag-critical' if r['severity'] == 'critical' else 'tag-warning' if r['severity'] == 'warning' else 'tag-good'
        reasons_html += f'<div class="reason-item">'
        reasons_html += f'<span class="reason-icon">{r["icon"]}</span>{r["text"]}'
        reasons_html += f'<span class="reason-tag {tag_class}">{r["tag"]}</span>'
        reasons_html += f'</div>'
        reasons_html += f'<div class="reason-detail">{r["detail"]}</div>'

    reasons_html += '</div>'
    st.markdown(reasons_html, unsafe_allow_html=True)

    st.markdown("---")

    # --- Section 2: Risk Score Breakdown ---
    st.subheader("2. Risk Score Component Breakdown")
    st.markdown("The final risk score is a weighted blend of 5 components:")

    components = {
        'Proxy Score (40%)': company_risk.get('comp_proxy', 0),
        'Linguistic GW Signal (15%)': company_risk.get('comp_linguistic', 0),
        'ESG-Controversy Gap (15%)': company_risk.get('comp_divergence', 0),
        'Low Claim Credibility (15%)': company_risk.get('comp_credibility_inv', 0),
        'Controversy Ratio (15%)': company_risk.get('comp_controversy_ratio', 0),
    }

    comp_df = pd.DataFrame({
        'Component': list(components.keys()),
        'Score (0-100)': [float(v) if pd.notna(v) else 0.0 for v in components.values()],
    })

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.bar(
            comp_df, x='Component', y='Score (0-100)',
            color='Score (0-100)', color_continuous_scale='RdYlGn_r',
            text='Score (0-100)',
            title=f'What drives {selected}\'s risk score?',
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(yaxis_range=[0, 110], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Risk"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 20], 'color': '#2ecc71'},
                    {'range': [20, 40], 'color': '#82e0aa'},
                    {'range': [40, 60], 'color': '#f4d03f'},
                    {'range': [60, 80], 'color': '#e67e22'},
                    {'range': [80, 100], 'color': '#e74c3c'},
                ],
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Section 3: ESG Pillar Analysis ---
    st.subheader("3. ESG Pillar Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Pillar scores bar chart
        pillar_data = {
            'Pillar': ['Environmental', 'Social', 'Governance', 'Total ESG', 'Controversy'],
            'Score': [
                float(company_fm.get('env_risk_score', 0)),
                float(company_fm.get('social_risk_score', 0)),
                float(company_fm.get('gov_risk_score', 0)),
                float(company_fm.get('total_esg_risk_score', 0)),
                float(company_fm.get('controversy_score', 0)),
            ],
            'Population Avg': [
                float(fm['env_risk_score'].mean()) if 'env_risk_score' in fm.columns else 0,
                float(fm['social_risk_score'].mean()) if 'social_risk_score' in fm.columns else 0,
                float(fm['gov_risk_score'].mean()) if 'gov_risk_score' in fm.columns else 0,
                float(fm['total_esg_risk_score'].mean()) if 'total_esg_risk_score' in fm.columns else 0,
                float(fm['controversy_score'].mean()) if 'controversy_score' in fm.columns else 0,
            ],
        }
        pillar_df = pd.DataFrame(pillar_data)

        fig = go.Figure()
        fig.add_trace(go.Bar(name=selected, x=pillar_df['Pillar'], y=pillar_df['Score'],
                             marker_color='#e74c3c'))
        fig.add_trace(go.Bar(name='Population Avg', x=pillar_df['Pillar'], y=pillar_df['Population Avg'],
                             marker_color='#3498db', opacity=0.6))
        fig.update_layout(barmode='group', title='ESG Scores vs Population Average',
                          yaxis_title='Score', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Radar chart
        categories = ['Environmental', 'Social', 'Governance', 'Controversy']
        cols_for_radar = ['env_risk_score', 'social_risk_score', 'gov_risk_score', 'controversy_score']

        company_vals = []
        avg_vals = []
        for col in cols_for_radar:
            if col in fm.columns:
                cmin, cmax = fm[col].min(), fm[col].max()
                company_vals.append(round((company_fm.get(col, 0) - cmin) / (cmax - cmin + 1e-8), 3))
                avg_vals.append(round((fm[col].mean() - cmin) / (cmax - cmin + 1e-8), 3))
            else:
                company_vals.append(0)
                avg_vals.append(0)

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=company_vals + [company_vals[0]],
            theta=categories + [categories[0]],
            fill='toself', name=selected,
            line=dict(color='#e74c3c'), opacity=0.7,
        ))
        fig.add_trace(go.Scatterpolar(
            r=avg_vals + [avg_vals[0]],
            theta=categories + [categories[0]],
            fill='toself', name='Population Avg',
            line=dict(color='#3498db', dash='dash'), opacity=0.4,
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title='ESG Profile Radar (normalized 0-1)', height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Pillar imbalance insight
    imbalance = company_fm.get('pillar_imbalance_score', 0)
    avg_imbalance = fm['pillar_imbalance_score'].mean() if 'pillar_imbalance_score' in fm.columns else 0
    if imbalance > avg_imbalance * 1.5:
        st.warning(f"Pillar Imbalance Score: **{imbalance:.2f}** (avg: {avg_imbalance:.2f}) -- "
                   f"This company has significantly uneven ESG performance across pillars, "
                   f"which can indicate selective reporting.")

    st.markdown("---")

    # --- Section 4: NLP & Linguistic Analysis ---
    st.subheader("4. NLP & Linguistic Analysis")
    st.markdown("How does this company's corporate text compare to population norms?")

    nlp_features = {
        'Greenwashing Signal Score': ('greenwashing_signal_score', 'Higher = more greenwashing language'),
        'Vague Language Count': ('vague_language_count', 'Vague terms like "committed to", "striving for"'),
        'Hedge Language Count': ('hedge_language_count', 'Hedging: "approximately", "potentially"'),
        'Superlative Count': ('superlative_count', '"Industry-leading", "world-class"'),
        'Future Language Count': ('future_language_count', '"Will", "plan to", "by 2030"'),
        'Concrete Evidence Count': ('concrete_evidence_count', '"Reduced by X%", "ISO certified"'),
        'Vague to Concrete Ratio': ('vague_to_concrete_ratio', 'Higher = more vague vs concrete'),
        'Text Sentiment Polarity': ('text_polarity', 'Positive bias in ESG text'),
        'Flesch Reading Ease': ('flesch_reading_ease', 'Lower = harder to read (obfuscation?)'),
        'Lexical Diversity': ('lexical_diversity', 'Unique words / total words'),
        'Aggregate ESG NLP Score': ('aggregate_esg_nlp_score', '0-100 composite ESG linguistic credibility'),
        'Regulatory Readiness': ('regulatory_readiness_score', 'Policy framework alignment (Paris, EU, TCFD, SDGs)'),
        'Policy-ESG Gap': ('policy_esg_gap', 'ESG talk without policy grounding = greenwashing'),
        'Narrative Credibility': ('narrative_credibility_index', 'Factual+strategic vs promotional+defensive'),
        'News GW Signal': ('news_greenwashing_signal', 'Promotional dominance + low credibility'),
        'Commitment Credibility': ('commitment_credibility_score', 'Past achievements + specific future targets'),
        'Temporal GW Signal': ('temporal_greenwashing_signal', 'Vague promises without past delivery'),
        'Multi-Signal GW Score': ('multi_signal_greenwashing_score', 'Ensemble of linguistic, news, temporal GW signals'),
    }

    nlp_rows = []
    for label, (col, desc) in nlp_features.items():
        val = company_fm.get(col, None)
        avg = fm[col].mean() if col in fm.columns else None
        if val is not None and avg is not None:
            diff = val - avg
            pct = (pd.Series(fm[col]).rank(pct=True).values[
                fm.index[fm['company_name'] == selected][0]] * 100) if col in fm.columns else 50
            nlp_rows.append({
                'Feature': label,
                'Value': round(float(val), 4),
                'Population Avg': round(float(avg), 4),
                'Difference': round(float(diff), 4),
                'Percentile': f"{pct:.0f}th",
                'Interpretation': desc,
            })

    if nlp_rows:
        nlp_df = pd.DataFrame(nlp_rows)

        # Horizontal bar chart comparing company vs average
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=nlp_df['Feature'], x=nlp_df['Value'], name=selected,
            orientation='h', marker_color='#e74c3c',
        ))
        fig.add_trace(go.Bar(
            y=nlp_df['Feature'], x=nlp_df['Population Avg'], name='Population Avg',
            orientation='h', marker_color='#3498db', opacity=0.6,
        ))
        fig.update_layout(
            barmode='group', title='NLP Features: Company vs Population',
            xaxis_title='Value', height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(nlp_df, use_container_width=True)

        # Linguistic verdict
        gw_signal = company_fm.get('greenwashing_signal_score', 0)
        vague = company_fm.get('vague_language_count', 0)
        concrete = company_fm.get('concrete_evidence_count', 0)
        if gw_signal > fm['greenwashing_signal_score'].quantile(0.75) if 'greenwashing_signal_score' in fm.columns else 0.5:
            st.error(f"LINGUISTIC RED FLAG: Greenwashing signal score ({gw_signal:.3f}) is in the "
                     f"top 25%. Uses {int(vague)} vague terms vs only {int(concrete)} concrete evidence points.")
        elif gw_signal > fm['greenwashing_signal_score'].median() if 'greenwashing_signal_score' in fm.columns else 0.5:
            st.warning(f"Linguistic caution: Greenwashing signal ({gw_signal:.3f}) is above median. "
                       f"Vague terms: {int(vague)}, Concrete evidence: {int(concrete)}.")
        else:
            st.success(f"Linguistic profile looks clean: GW signal ({gw_signal:.3f}) is below median. "
                       f"Concrete evidence ({int(concrete)}) outweighs vague language ({int(vague)}).")

    st.markdown("---")

    # --- Section 5: Key Greenwashing Indicators ---
    st.subheader("5. Key Greenwashing Indicators")

    indicators = {
        'Controversy-Risk Ratio': ('controversy_risk_ratio',
            'Controversy relative to ESG risk. High = controversies not reflected in ESG score'),
        'ESG-Controversy Divergence': ('esg_controversy_divergence',
            'z(controversy) - z(ESG). High = claims low risk but has high controversy'),
        'Risk-Controversy Mismatch': ('risk_controversy_mismatch',
            'Binary flag: ESG risk and controversy levels are inconsistent'),
        'Combined Anomaly Score': ('combined_anomaly_score',
            'Statistical anomaly detection across multiple features'),
        'Pillar Imbalance Score': ('pillar_imbalance_score',
            'Std dev across E/S/G pillars. High = uneven performance'),
        'ESG Sector Z-Score': ('esg_sector_zscore',
            'How this company compares to its sector peers'),
    }

    indicator_rows = []
    for label, (col, desc) in indicators.items():
        val = company_fm.get(col, None)
        if val is not None and col in fm.columns:
            q75 = fm[col].quantile(0.75)
            status = "HIGH" if float(val) > float(q75) else "Normal"
            indicator_rows.append({
                'Indicator': label,
                'Value': round(float(val), 4),
                'Threshold (75th pct)': round(float(q75), 4),
                'Status': status,
                'Description': desc,
            })

    if indicator_rows:
        ind_df = pd.DataFrame(indicator_rows)
        # Color the status column
        def color_status(val):
            if val == 'HIGH':
                return 'background-color: #ffcccc; color: red; font-weight: bold'
            return 'background-color: #ccffcc; color: green'

        st.dataframe(ind_df.style.applymap(color_status, subset=['Status']),
                     use_container_width=True)

        high_count = sum(1 for r in indicator_rows if r['Status'] == 'HIGH')
        st.markdown(f"**{high_count} out of {len(indicator_rows)} indicators** are above the 75th percentile threshold.")

    st.markdown("---")

    # --- Section 6: Sector Peer Comparison ---
    st.subheader("6. Sector Peer Comparison")
    sector = company_risk.get('sector', None)
    if sector:
        peers = df[df['sector'] == sector].copy().sort_values('risk_score', ascending=False)
        peers['highlight'] = peers['company_name'] == selected
        peer_rank = list(peers['company_name']).index(selected) + 1

        st.markdown(f"**{selected}** ranks **#{peer_rank} out of {len(peers)}** "
                    f"in the **{sector}** sector by risk score.")

        fig = px.bar(
            peers, x='company_name', y='risk_score',
            color='highlight',
            color_discrete_map={True: '#e74c3c', False: '#bdc3c7'},
            title=f'Risk Scores in {sector} Sector',
            labels={'risk_score': 'Risk Score', 'company_name': 'Company'},
            hover_data=['risk_tier'],
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Sector stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sector Avg Risk", f"{peers['risk_score'].mean():.1f}")
        with col2:
            st.metric("Company Risk", f"{risk_score:.1f}",
                      delta=f"{risk_score - peers['risk_score'].mean():.1f} vs sector avg",
                      delta_color="inverse")
        with col3:
            st.metric("Sector Rank", f"#{peer_rank}/{len(peers)}")

    st.markdown("---")

    # --- Section 7: Model Prediction Details ---
    st.subheader("7. Model Prediction Summary")
    if not pred.empty:
        pred_row = pred[pred['company_name'] == selected]
        if not pred_row.empty:
            pred_row = pred_row.iloc[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                proxy = pred_row.get('gw_proxy_score', 0)
                st.metric("Proxy Score (0-5)", f"{int(proxy)}/5")
            with col2:
                label = pred_row.get('gw_label_binary', 0)
                st.metric("Binary Label", "FLAGGED" if label == 1 else "NOT FLAGGED")
            with col3:
                st.metric("ESG Risk", f"{pred_row.get('total_esg_risk_score', 'N/A')}")

            if int(proxy) >= 2:
                st.error(f"This company triggered **{int(proxy)} out of 5** greenwashing indicators "
                         f"and is classified as a **greenwashing risk**.")
            else:
                st.success(f"This company triggered only **{int(proxy)} out of 5** greenwashing indicators "
                           f"and is classified as **low risk**.")

    # --- Section 8: Full Feature Profile (expandable) ---
    with st.expander("View Full Feature Profile (all 161 features)", expanded=False):
        all_features = {}
        for col in fm.columns:
            if col not in ['symbol', 'company_name', 'sector', 'industry', 'description', 'source']:
                val = company_fm.get(col)
                if val is not None:
                    try:
                        if not pd.isna(val):
                            all_features[col] = round(float(val), 6)
                    except (ValueError, TypeError):
                        all_features[col] = str(val)

        full_df = pd.DataFrame({
            'Feature': list(all_features.keys()),
            'Value': list(all_features.values()),
        })
        st.dataframe(full_df, use_container_width=True, height=600)


# ============================================================================
# PAGE 6: REAL-TIME INTELLIGENCE
# ============================================================================

# --- Sentiment Analyzer (reused from project's NLP module) ---
class _LiveSentimentAnalyzer:
    """Lightweight sentiment analyzer for live news headlines."""

    POSITIVE = {
        'sustainable': 0.9, 'sustainability': 0.9, 'renewable': 0.9,
        'innovation': 0.8, 'excellent': 0.9, 'leading': 0.7, 'growth': 0.7,
        'efficient': 0.7, 'clean': 0.8, 'green': 0.8, 'responsible': 0.8,
        'transparency': 0.8, 'ethical': 0.8, 'diverse': 0.7, 'improvement': 0.7,
        'committed': 0.7, 'progress': 0.7, 'achievement': 0.8, 'certified': 0.7,
        'award': 0.7, 'strong': 0.6, 'resilient': 0.7, 'positive': 0.7,
        'success': 0.7, 'protect': 0.7, 'conservation': 0.8, 'recycle': 0.8,
        'net-zero': 0.9, 'carbon-neutral': 0.9, 'upgrade': 0.7, 'outperform': 0.8,
        'profit': 0.6, 'partnership': 0.6, 'launch': 0.5, 'invest': 0.6,
    }
    NEGATIVE = {
        'pollution': -0.9, 'violation': -0.9, 'penalty': -0.8, 'fine': -0.7,
        'fined': -0.8, 'lawsuit': -0.8, 'scandal': -0.9, 'fraud': -0.9,
        'corruption': -0.9, 'controversy': -0.8, 'risk': -0.5, 'toxic': -0.9,
        'unsafe': -0.8, 'harmful': -0.8, 'damage': -0.7, 'spill': -0.8,
        'accident': -0.7, 'failure': -0.7, 'decline': -0.6, 'loss': -0.6,
        'layoff': -0.7, 'discrimination': -0.9, 'harassment': -0.9,
        'greenwashing': -0.9, 'misleading': -0.8, 'deceptive': -0.9,
        'illegal': -0.9, 'misconduct': -0.8, 'bribery': -0.9, 'crash': -0.8,
        'downgrade': -0.7, 'investigation': -0.7, 'probe': -0.7, 'recall': -0.6,
        'warning': -0.6, 'threat': -0.7, 'debt': -0.5, 'bankruptcy': -0.9,
        'shutdown': -0.7, 'protest': -0.7, 'strike': -0.6, 'emission': -0.5,
    }

    @staticmethod
    def score(text):
        if not isinstance(text, str):
            return 0.0
        words = text.lower().split()
        scores = []
        for w in words:
            w_clean = re.sub(r'[^a-z\-]', '', w)
            if w_clean in _LiveSentimentAnalyzer.POSITIVE:
                scores.append(_LiveSentimentAnalyzer.POSITIVE[w_clean])
            elif w_clean in _LiveSentimentAnalyzer.NEGATIVE:
                scores.append(_LiveSentimentAnalyzer.NEGATIVE[w_clean])
        if not scores:
            return 0.0
        raw = sum(scores)
        return round(raw / np.sqrt(raw ** 2 + 15), 4)


def _fetch_news_google_rss(company_name, num_articles=15):
    """Fetch live news from Google News RSS feed (free, no API key)."""
    try:
        import urllib.request
        query = quote(f"{company_name} ESG sustainability")
        url = f"https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_data = resp.read().decode('utf-8')

        root = ET.fromstring(xml_data)
        articles = []
        for item in root.iter('item'):
            title = item.find('title')
            pub_date = item.find('pubDate')
            source = item.find('source')
            link = item.find('link')

            title_text = title.text if title is not None else ''
            # Clean HTML entities from Google News titles
            title_text = re.sub(r'<[^>]+>', '', title_text)

            pub_str = pub_date.text if pub_date is not None else ''
            source_name = source.text if source is not None else 'Unknown'
            link_url = link.text if link is not None else ''

            # Parse date
            try:
                dt = datetime.strptime(pub_str[:25], '%a, %d %b %Y %H:%M:%S')
            except Exception:
                dt = datetime.now()

            articles.append({
                'title': title_text,
                'source': source_name,
                'published': dt,
                'url': link_url,
                'age_hours': (datetime.now() - dt).total_seconds() / 3600,
            })
            if len(articles) >= num_articles:
                break

        return articles
    except Exception as e:
        return []


def _compute_live_risk_delta(news_articles, base_risk_score):
    """Compute risk score adjustment based on live news sentiment."""
    if not news_articles:
        return 0.0, 0.0, []

    scored = []
    for art in news_articles:
        sent = _LiveSentimentAnalyzer.score(art['title'])
        # Recent news has more weight (exponential decay)
        recency_weight = np.exp(-art['age_hours'] / 72)  # 72h half-life
        weighted_sent = sent * recency_weight
        scored.append({
            **art,
            'sentiment': sent,
            'recency_weight': round(recency_weight, 3),
            'weighted_sentiment': round(weighted_sent, 4),
        })

    avg_sentiment = np.mean([s['weighted_sentiment'] for s in scored])
    # Convert sentiment (-1 to +1) to risk delta (-15 to +15)
    risk_delta = round(-avg_sentiment * 15, 1)
    new_risk = max(0, min(100, base_risk_score + risk_delta))
    return risk_delta, new_risk, scored


def page_realtime_intelligence(data):
    """Real-Time Intelligence page -- live news, sentiment, breaking risk alerts."""

    # --- Dark terminal-style header ---
    st.markdown("""
    <style>
    .terminal-header {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        padding: 25px; border-radius: 12px; margin-bottom: 20px;
        border: 1px solid #0f3460;
    }
    .terminal-header h1 { color: #00ff88; margin: 0; font-family: monospace; }
    .terminal-header p { color: #7ec8e3; margin: 5px 0 0 0; font-family: monospace; font-size: 14px; }
    .alert-critical {
        background: linear-gradient(90deg, #ff0000 0%, #cc0000 100%);
        color: white; padding: 12px 20px; border-radius: 8px; margin: 5px 0;
        font-weight: bold; font-family: monospace; animation: pulse 2s infinite;
    }
    .alert-warning {
        background: linear-gradient(90deg, #ff8c00 0%, #cc7000 100%);
        color: white; padding: 12px 20px; border-radius: 8px; margin: 5px 0;
        font-weight: bold; font-family: monospace;
    }
    .alert-positive {
        background: linear-gradient(90deg, #00aa44 0%, #008833 100%);
        color: white; padding: 12px 20px; border-radius: 8px; margin: 5px 0;
        font-weight: bold; font-family: monospace;
    }
    .news-card {
        background: #f8f9fa; padding: 12px 16px; border-radius: 8px;
        margin: 6px 0; border-left: 4px solid #3498db;
    }
    .news-negative { border-left-color: #e74c3c; }
    .news-positive { border-left-color: #2ecc71; }
    .news-neutral { border-left-color: #95a5a6; }
    .risk-ticker {
        background: #1a1a2e; color: #00ff88; padding: 8px 16px;
        border-radius: 6px; font-family: monospace; font-size: 13px;
        display: inline-block; margin: 3px;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.85; } }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="terminal-header">
        <h1>REAL-TIME ESG INTELLIGENCE TERMINAL</h1>
        <p>Live news monitoring | Sentiment analysis | Breaking risk alerts</p>
    </div>
    """, unsafe_allow_html=True)

    df = data['risk_scores']
    if df.empty:
        st.warning("Risk scores not available. Run `python model_pipeline.py` first.")
        return

    # --- Company selector ---
    company_names = sorted(df['company_name'].dropna().unique().tolist())
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox("Select Company for Live Monitoring", company_names,
                                index=0, key='rt_company')
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh = st.button("Refresh Live Data", type="primary")

    if not selected:
        return

    company = df[df['company_name'] == selected].iloc[0]
    base_risk = float(company['risk_score'])

    st.markdown("---")

    # --- Fetch live news ---
    with st.spinner(f"Fetching live news for {selected}..."):
        articles = _fetch_news_google_rss(selected)

    if not articles:
        st.warning(f"Could not fetch live news for **{selected}**. "
                   f"This may be due to network restrictions. Showing analysis with simulated data.")
        # Generate simulated news for demo purposes
        np.random.seed(hash(selected) % 2**31)
        sim_headlines = [
            f"{selected} announces new sustainability initiative targeting carbon reduction",
            f"ESG rating agency upgrades {selected} environmental score",
            f"{selected} faces scrutiny over supply chain labor practices",
            f"Investors question {selected}'s green bond transparency",
            f"{selected} partners with renewable energy provider for operations",
            f"Report highlights {selected}'s progress on diversity targets",
            f"{selected} CEO defends ESG strategy amid shareholder concerns",
            f"Environmental group praises {selected}'s waste reduction program",
            f"{selected} under investigation for emissions data discrepancies",
            f"Analysts see strong ESG momentum for {selected} in 2026",
        ]
        articles = [{
            'title': h,
            'source': np.random.choice(['Reuters', 'Bloomberg', 'CNBC', 'Financial Times', 'WSJ']),
            'published': datetime.now() - timedelta(hours=np.random.randint(1, 120)),
            'url': '',
            'age_hours': float(np.random.randint(1, 120)),
        } for h in sim_headlines]

    # --- Compute live risk adjustment ---
    risk_delta, new_risk, scored_articles = _compute_live_risk_delta(articles, base_risk)

    # === SECTION 1: BREAKING RISK ALERTS ===
    st.subheader("Breaking Risk Alerts")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Base Risk Score", f"{base_risk:.1f}")
    with col2:
        delta_str = f"+{risk_delta:.1f}" if risk_delta > 0 else f"{risk_delta:.1f}"
        st.metric("Live Adjustment", delta_str,
                  delta=f"{risk_delta:.1f} from news",
                  delta_color="inverse")
    with col3:
        st.metric("Adjusted Risk", f"{new_risk:.1f}")
    with col4:
        neg_count = sum(1 for a in scored_articles if a['sentiment'] < -0.1)
        st.metric("Negative Signals", f"{neg_count}/{len(scored_articles)}")

    # Risk change alert
    if abs(risk_delta) >= 5:
        if risk_delta > 0:
            st.markdown(
                f'<div class="alert-critical">'
                f'ALERT: {selected} risk jumped from {base_risk:.0f} -> {new_risk:.0f} '
                f'(+{risk_delta:.1f}) due to recent negative ESG news coverage</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="alert-positive">'
                f'POSITIVE: {selected} risk improved from {base_risk:.0f} -> {new_risk:.0f} '
                f'({risk_delta:.1f}) due to favorable ESG news sentiment</div>',
                unsafe_allow_html=True)
    elif abs(risk_delta) >= 2:
        st.markdown(
            f'<div class="alert-warning">'
            f'WATCH: {selected} risk shifted from {base_risk:.0f} -> {new_risk:.0f} '
            f'({delta_str}) -- moderate news activity detected</div>',
            unsafe_allow_html=True)
    else:
        st.info(f"Risk stable for {selected}. No significant sentiment shifts detected in recent news.")

    st.markdown("---")

    # === SECTION 2: LIVE RISK GAUGE (Before/After) ===
    st.subheader("Live Risk Comparison")
    col1, col2 = st.columns(2)

    for col, (title, value, color) in zip(
        [col1, col2],
        [("Base Risk (Model)", base_risk, "royalblue"),
         ("Adjusted Risk (Live)", new_risk, "darkred")]
    ):
        with col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=value,
                delta={'reference': base_risk, 'increasing': {'color': 'red'},
                       'decreasing': {'color': 'green'}},
                title={'text': title},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 20], 'color': '#2ecc71'},
                        {'range': [20, 40], 'color': '#82e0aa'},
                        {'range': [40, 60], 'color': '#f4d03f'},
                        {'range': [60, 80], 'color': '#e67e22'},
                        {'range': [80, 100], 'color': '#e74c3c'},
                    ],
                }
            ))
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # === SECTION 3: NEWS SENTIMENT TIMELINE ===
    st.subheader("News Sentiment Timeline")

    if scored_articles:
        news_df = pd.DataFrame(scored_articles)
        news_df = news_df.sort_values('published')

        # Sentiment over time
        fig = go.Figure()
        colors = ['#e74c3c' if s < -0.1 else '#2ecc71' if s > 0.1 else '#95a5a6'
                  for s in news_df['sentiment']]

        fig.add_trace(go.Scatter(
            x=news_df['published'], y=news_df['sentiment'],
            mode='markers+lines',
            marker=dict(size=12, color=colors, line=dict(width=1, color='white')),
            line=dict(color='#7f8c8d', width=1),
            text=news_df['title'],
            hovertemplate='<b>%{text}</b><br>Sentiment: %{y:.3f}<br>Date: %{x}<extra></extra>',
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hrect(y0=-1, y1=-0.1, fillcolor="red", opacity=0.05)
        fig.add_hrect(y0=0.1, y1=1, fillcolor="green", opacity=0.05)
        fig.update_layout(
            title=f'Sentiment of Recent News -- {selected}',
            xaxis_title='Date', yaxis_title='Sentiment Score',
            yaxis_range=[-1, 1], height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Sentiment distribution
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                news_df, x='sentiment', nbins=15,
                title='Sentiment Distribution',
                color_discrete_sequence=['#3498db'],
                labels={'sentiment': 'Sentiment Score', 'count': 'Articles'},
            )
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Sentiment pie
            sent_cats = pd.Series(['Negative' if s < -0.1 else 'Positive' if s > 0.1 else 'Neutral'
                                   for s in news_df['sentiment']])
            sent_counts = sent_cats.value_counts().reset_index()
            sent_counts.columns = ['Sentiment', 'Count']
            fig = px.pie(sent_counts, values='Count', names='Sentiment',
                         title='Sentiment Breakdown',
                         color='Sentiment',
                         color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c',
                                             'Neutral': '#95a5a6'},
                         hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # === SECTION 4: LIVE NEWS FEED ===
    st.subheader("Live News Feed")

    if scored_articles:
        for art in sorted(scored_articles, key=lambda x: x['published'], reverse=True):
            sent = art['sentiment']
            if sent < -0.1:
                css_class = 'news-negative'
                badge = '🔴 NEGATIVE'
                badge_color = '#e74c3c'
            elif sent > 0.1:
                css_class = 'news-positive'
                badge = '🟢 POSITIVE'
                badge_color = '#2ecc71'
            else:
                css_class = 'news-neutral'
                badge = '⚪ NEUTRAL'
                badge_color = '#95a5a6'

            age = art['age_hours']
            if age < 1:
                age_str = f"{int(age * 60)}m ago"
            elif age < 24:
                age_str = f"{int(age)}h ago"
            else:
                age_str = f"{int(age / 24)}d ago"

            st.markdown(
                f'<div class="news-card {css_class}">'
                f'<span style="color:{badge_color};font-weight:bold;">{badge}</span> '
                f'<span style="color:#666;font-size:12px;">| {art["source"]} | {age_str} '
                f'| Sentiment: {sent:+.3f}</span><br>'
                f'<b>{art["title"]}</b></div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # === SECTION 5: MULTI-COMPANY RISK TICKER ===
    st.subheader("Portfolio Risk Monitor")
    st.markdown("Select multiple companies to compare live risk adjustments:")

    multi_select = st.multiselect(
        "Select companies for monitoring",
        company_names,
        default=company_names[:5] if len(company_names) >= 5 else company_names,
        key='rt_multi',
    )

    if multi_select and st.button("Scan Selected Companies", key='rt_scan'):
        ticker_data = []
        progress = st.progress(0)
        status = st.empty()

        for i, comp in enumerate(multi_select):
            status.text(f"Scanning {comp}... ({i+1}/{len(multi_select)})")
            progress.progress((i + 1) / len(multi_select))

            comp_row = df[df['company_name'] == comp]
            if comp_row.empty:
                continue
            comp_risk = float(comp_row.iloc[0]['risk_score'])
            comp_articles = _fetch_news_google_rss(comp, num_articles=8)

            if not comp_articles:
                # Simulate for demo
                np.random.seed(hash(comp) % 2**31)
                comp_articles = [{
                    'title': f"{comp} ESG news headline {j}",
                    'source': 'Simulated',
                    'published': datetime.now() - timedelta(hours=np.random.randint(1, 72)),
                    'url': '',
                    'age_hours': float(np.random.randint(1, 72)),
                } for j in range(5)]

            delta, adjusted, _ = _compute_live_risk_delta(comp_articles, comp_risk)
            ticker_data.append({
                'Company': comp,
                'Sector': comp_row.iloc[0].get('sector', 'N/A'),
                'Base Risk': comp_risk,
                'Live Delta': delta,
                'Adjusted Risk': adjusted,
                'News Count': len(comp_articles),
                'Status': 'ALERT' if delta > 5 else 'WATCH' if delta > 2 else
                          'IMPROVED' if delta < -2 else 'STABLE',
            })

        progress.empty()
        status.empty()

        if ticker_data:
            ticker_df = pd.DataFrame(ticker_data).sort_values('Live Delta', ascending=False)

            # Risk change bar chart
            fig = px.bar(
                ticker_df, x='Company', y='Live Delta',
                color='Live Delta',
                color_continuous_scale='RdYlGn_r',
                title='Live Risk Adjustments Across Portfolio',
                text='Live Delta',
                hover_data=['Base Risk', 'Adjusted Risk', 'Status'],
            )
            fig.update_traces(texttemplate='%{text:+.1f}', textposition='outside')
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=450, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            # Ticker display
            for _, row in ticker_df.iterrows():
                if row['Status'] == 'ALERT':
                    color = '#ff4444'
                    arrow = '▲'
                elif row['Status'] == 'WATCH':
                    color = '#ffaa00'
                    arrow = '▲'
                elif row['Status'] == 'IMPROVED':
                    color = '#00cc44'
                    arrow = '▼'
                else:
                    color = '#888888'
                    arrow = '—'

                st.markdown(
                    f'<span class="risk-ticker">'
                    f'<span style="color:{color};font-weight:bold;">{arrow} {row["Company"]}</span> '
                    f'{row["Base Risk"]:.0f} → {row["Adjusted Risk"]:.0f} '
                    f'(<span style="color:{color}">{row["Live Delta"]:+.1f}</span>) '
                    f'[{row["Status"]}]</span>',
                    unsafe_allow_html=True)

            # Summary table
            st.dataframe(ticker_df, use_container_width=True)

            # Alert summary
            alerts = ticker_df[ticker_df['Status'] == 'ALERT']
            if not alerts.empty:
                st.error(f"**{len(alerts)} company(s) require immediate attention** due to "
                         f"significant negative ESG news sentiment.")


# ============================================================================
# PAGE 7: LLM-POWERED ESG REPORT ANALYZER
# ============================================================================

def _extract_pdf_text(uploaded_file):
    """Extract text from uploaded PDF file. Tries multiple methods."""

    # Method 1: PyPDF2 (fast, works for text-based PDFs)
    try:
        from PyPDF2 import PdfReader
        uploaded_file.seek(0)
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        text = text.strip()
        if len(text) >= 50:
            return text
    except Exception:
        pass

    # Method 2: PyMuPDF/fitz (better extraction for complex PDFs)
    try:
        import fitz  # PyMuPDF
        uploaded_file.seek(0)
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        text = text.strip()
        if len(text) >= 50:
            return text
    except ImportError:
        pass
    except Exception:
        pass

    # If both methods fail, return None (caller will use Gemini Vision)
    return None


def _extract_pdf_with_gemini(uploaded_file, api_key):
    """Use Gemini Vision to extract text from scanned/image-based PDFs."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        uploaded_file.seek(0)
        pdf_bytes = uploaded_file.read()

        # Upload file to Gemini
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        gemini_file = genai.upload_file(tmp_path, mime_type='application/pdf')

        model = genai.GenerativeModel('gemini-2.0-flash')

        # Retry up to 2 times for rate limits
        import time
        response = None
        for attempt in range(3):
            try:
                response = model.generate_content(
                    [
                        gemini_file,
                        "Extract ALL text from this PDF document. Return the complete text content exactly as it appears. "
                        "If it contains tables, preserve the structure. If it contains images with text, extract that text too. "
                        "Return ONLY the extracted text, nothing else."
                    ],
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=8000,
                        temperature=0.1,
                    )
                )
                break
            except Exception as retry_err:
                if "429" in str(retry_err) and attempt < 2:
                    time.sleep(30)  # Wait 30s before retry
                    continue
                raise

        if response is None:
            return None

        # Cleanup
        try:
            os.unlink(tmp_path)
            gemini_file.delete()
        except Exception:
            pass

        text = response.text.strip()
        return text if len(text) >= 20 else None

    except Exception as e:
        st.error(f"Gemini Vision error: {str(e)[:200]}")
        if "429" in str(e) or "quota" in str(e).lower():
            st.warning("API rate limit reached. Wait 1 minute and try again, or try tomorrow for daily limits.")
        return None


def _call_gemini(api_key, prompt, max_tokens=8000):
    """Call Google Gemini API for ESG report analysis."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.3,
            )
        )
        return response.text
    except Exception as e:
        return f"ERROR: {str(e)}"


def _build_analysis_prompt(report_text, company_data=None):
    """Build the Gemini prompt for ESG greenwashing analysis."""

    company_context = ""
    if company_data is not None:
        company_context = f"""
EXISTING DATA FOR THIS COMPANY (from our ML analysis):
- ESG Risk Score: {company_data.get('total_esg_risk_score', 'N/A')}
- Controversy Score: {company_data.get('controversy_score', 'N/A')}
- Greenwashing Risk Score: {company_data.get('risk_score', 'N/A')}/100
- Risk Tier: {company_data.get('risk_tier', 'N/A')}
- Sector: {company_data.get('sector', 'N/A')}

Use this data to cross-check claims in the report. Flag contradictions.
"""

    prompt = f"""You are an expert ESG analyst specializing in greenwashing detection.
Analyze the following ESG/sustainability report text and identify greenwashing risks.

{company_context}

REPORT TEXT:
{report_text[:15000]}

INSTRUCTIONS - Respond in EXACTLY this JSON format (no markdown, no code fences):
{{
  "company_name": "detected company name or Unknown",
  "overall_risk": "HIGH/MEDIUM/LOW",
  "overall_score": <number 0-100>,
  "summary": "2-3 sentence overall assessment",
  "claims": [
    {{
      "claim_text": "exact quote from report",
      "category": "Environmental/Social/Governance",
      "risk_level": "HIGH/MEDIUM/LOW",
      "issue": "specific greenwashing concern",
      "explanation": "detailed explanation of why this is suspicious or credible",
      "evidence_type": "Vague Promise/Missing Data/Contradicts Data/Unverified/Credible"
    }}
  ],
  "red_flags": ["list of major concerns"],
  "positive_signals": ["list of credible practices"],
  "recommendations": ["list of what the company should improve"]
}}

ANALYSIS RULES:
1. Extract 8-15 specific claims from the report
2. For each claim, assess if it shows greenwashing patterns:
   - Vague language without measurable targets ("committed to sustainability")
   - Future promises without past performance data ("we will reduce by 2030")
   - Cherry-picked metrics that hide overall poor performance
   - Superlatives without evidence ("industry-leading", "world-class")
   - Missing third-party verification for bold claims
3. Cross-reference with company data if available
4. Rate overall greenwashing risk 0-100
5. Be specific in explanations -- cite exact words from the report

Return ONLY valid JSON, no other text."""

    return prompt


def _parse_gemini_response(response_text):
    """Parse the JSON response from Gemini."""
    try:
        # Clean response -- remove markdown code fences if present
        cleaned = response_text.strip()
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        match = re.search(r'\{[\s\S]*\}', response_text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None


def _fallback_analysis(text):
    """Rule-based fallback analysis when Gemini is unavailable."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    vague_patterns = [
        r'committed to', r'striving for', r'working towards', r'aim to',
        r'plan to', r'intend to', r'aspire', r'endeavour', r'dedicated to',
        r'believe in', r'passionate about', r'focused on',
    ]
    superlative_patterns = [
        r'industry.?leading', r'world.?class', r'best.?in.?class', r'leading',
        r'top.?tier', r'premier', r'unmatched', r'unparalleled',
    ]
    future_patterns = [
        r'will\s', r'by 20\d\d', r'going forward', r'in the future',
        r'upcoming', r'planned', r'roadmap', r'target of',
    ]
    concrete_patterns = [
        r'\d+%', r'\d+ ton', r'ISO \d+', r'GRI', r'TCFD', r'verified by',
        r'audited', r'certified', r'reduced by', r'achieved',
    ]

    claims = []
    for sent in sentences[:30]:
        sent_lower = sent.lower()
        is_vague = any(re.search(p, sent_lower) for p in vague_patterns)
        is_superlative = any(re.search(p, sent_lower) for p in superlative_patterns)
        is_future = any(re.search(p, sent_lower) for p in future_patterns)
        is_concrete = any(re.search(p, sent_lower) for p in concrete_patterns)

        if is_vague or is_superlative or is_future or is_concrete:
            if is_vague and not is_concrete:
                risk = "HIGH"
                issue = "Uses vague language without measurable commitments"
                evidence = "Vague Promise"
            elif is_superlative:
                risk = "MEDIUM"
                issue = "Superlative claims without supporting evidence"
                evidence = "Unverified"
            elif is_future and not is_concrete:
                risk = "MEDIUM"
                issue = "Future promise without past performance data"
                evidence = "Vague Promise"
            elif is_concrete:
                risk = "LOW"
                issue = "Contains measurable data points"
                evidence = "Credible"
            else:
                risk = "MEDIUM"
                issue = "Requires further verification"
                evidence = "Unverified"

            claims.append({
                'claim_text': sent[:200],
                'category': 'Environmental' if any(w in sent_lower for w in
                    ['carbon', 'emission', 'climate', 'renewable', 'energy', 'water', 'waste'])
                    else 'Social' if any(w in sent_lower for w in
                    ['employee', 'diversity', 'community', 'safety', 'human rights'])
                    else 'Governance',
                'risk_level': risk,
                'issue': issue,
                'explanation': f"This statement {'uses vague aspirational language' if is_vague else 'contains superlatives' if is_superlative else 'makes future promises' if is_future else 'provides concrete data'}. "
                    + ("No specific metrics or timelines are mentioned." if risk != "LOW" else "Includes verifiable data points."),
                'evidence_type': evidence,
            })

    high_count = sum(1 for c in claims if c['risk_level'] == 'HIGH')
    total = len(claims) if claims else 1
    score = min(100, int((high_count / total) * 100 + 20))

    return {
        'company_name': 'Unknown (PDF Analysis)',
        'overall_risk': 'HIGH' if score >= 60 else 'MEDIUM' if score >= 40 else 'LOW',
        'overall_score': score,
        'summary': f'Analyzed {len(sentences)} sentences, found {len(claims)} ESG claims. '
                   f'{high_count} show high greenwashing risk patterns.',
        'claims': claims[:15],
        'red_flags': [c['issue'] for c in claims if c['risk_level'] == 'HIGH'][:5],
        'positive_signals': [c['issue'] for c in claims if c['risk_level'] == 'LOW'][:5],
        'recommendations': [
            'Add measurable targets with specific timelines',
            'Include third-party verification for major claims',
            'Report on past performance, not just future intentions',
        ],
    }


def page_esg_report_analyzer(data):
    """LLM-Powered ESG Report Analyzer -- Upload, Analyze, Explain."""

    # --- Styled header ---
    st.markdown("""
    <style>
    .analyzer-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 25px; border-radius: 12px; margin-bottom: 20px;
        border: 1px solid #e94560;
    }
    .analyzer-header h1 { color: #e94560; margin: 0; font-family: monospace; }
    .analyzer-header p { color: #7ec8e3; margin: 5px 0 0 0; font-size: 14px; }
    .claim-high {
        background: #fff0f0; padding: 15px; border-radius: 8px; margin: 8px 0;
        border-left: 5px solid #e74c3c;
    }
    .claim-medium {
        background: #fff8e1; padding: 15px; border-radius: 8px; margin: 8px 0;
        border-left: 5px solid #ff9800;
    }
    .claim-low {
        background: #f0fff0; padding: 15px; border-radius: 8px; margin: 8px 0;
        border-left: 5px solid #4caf50;
    }
    .claim-label {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 12px; font-weight: bold; color: white; margin-right: 8px;
    }
    .label-high { background: #e74c3c; }
    .label-medium { background: #ff9800; }
    .label-low { background: #4caf50; }
    .red-flag-box {
        background: #1a1a2e; color: #ff4444; padding: 12px 16px;
        border-radius: 8px; margin: 4px 0; font-family: monospace;
        border: 1px solid #333;
    }
    .positive-box {
        background: #1a1a2e; color: #00cc66; padding: 12px 16px;
        border-radius: 8px; margin: 4px 0; font-family: monospace;
        border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="analyzer-header">
        <h1>LLM-POWERED ESG REPORT ANALYZER</h1>
        <p>Upload PDF → AI Extracts Claims → Detects Greenwashing → Explains Why</p>
        <p>Powered by Google Gemini 2.0 Flash</p>
    </div>
    """, unsafe_allow_html=True)

    # --- API Key (auto-loaded from .env) ---
    env_key = os.environ.get('GEMINI_API_KEY', '')
    if env_key:
        api_key = env_key
        st.success("Gemini API key loaded from .env file")
        use_fallback = False
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            api_key = st.text_input(
                "Google Gemini API Key",
                type="password",
                placeholder="Paste your Gemini API key here...",
                help="Get free key at https://aistudio.google.com/apikey",
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            use_fallback = st.checkbox("Use without API key", value=False,
                                       help="Uses rule-based analysis instead of LLM")

        if not api_key and not use_fallback:
            st.info("Enter your Gemini API key above, or check 'Use without API key' for rule-based analysis.")
            return

    st.markdown("---")

    # --- PDF Upload ---
    st.subheader("Upload ESG / Sustainability Report")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Drop a PDF report here",
            type=['pdf'],
            help="Upload any ESG, sustainability, or annual report (PDF format)",
        )

    with col2:
        # Optional: match with existing company
        df = data.get('risk_scores', pd.DataFrame())
        company_match = None
        if not df.empty:
            company_names = ['Auto-detect'] + sorted(df['company_name'].dropna().unique().tolist())
            selected_company = st.selectbox(
                "Match with existing company (optional)",
                company_names,
                help="Select if this report belongs to a company in our database",
            )
            if selected_company != 'Auto-detect':
                company_match = df[df['company_name'] == selected_company].iloc[0].to_dict()

    if not uploaded_file:
        st.markdown("---")
        st.markdown("### What this analyzer does:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **1. Extract Claims**
            - Finds all ESG claims in the report
            - Categorizes as Environmental, Social, or Governance
            - Identifies specific commitments and targets
            """)
        with col2:
            st.markdown("""
            **2. Detect Greenwashing**
            - Flags vague promises without data
            - Spots superlatives without evidence
            - Finds future-only claims with no track record
            - Cross-checks with real controversy data
            """)
        with col3:
            st.markdown("""
            **3. Explain Everything**
            - Color-coded risk for each claim
            - Plain-English explanations
            - Actionable recommendations
            - Overall greenwashing risk score
            """)
        return

    # --- Extract text from PDF (multi-method) ---
    with st.spinner("Extracting text from PDF..."):
        report_text = _extract_pdf_text(uploaded_file)

    if not report_text:
        # Fallback: Use Gemini Vision for scanned/image PDFs
        if api_key and not use_fallback:
            st.warning("Standard text extraction failed. Using Gemini AI Vision to read the scanned PDF...")
            with st.spinner("Gemini AI is reading the scanned PDF... (this may take 15-30 seconds)"):
                report_text = _extract_pdf_with_gemini(uploaded_file, api_key)

            if report_text:
                st.success(f"Gemini Vision extracted **{len(report_text):,} characters** from scanned PDF")
            else:
                st.error("Gemini Vision could not extract text from this PDF either. The file may be corrupted or empty.")
                return
        else:
            st.error("Could not extract text from PDF. The file may be scanned/image-based.")
            st.markdown("""
            **Tips:**
            - Enter a Gemini API key above -- it can read scanned/image PDFs using AI Vision
            - Or upload a **text-based PDF** (not a screenshot)
            - ESG/sustainability reports from company websites are usually text-based
            """)
            return
    else:
        st.success(f"Extracted **{len(report_text):,} characters** from {uploaded_file.name}")

    # Show extracted text preview
    with st.expander("Preview extracted text", expanded=False):
        st.text(report_text[:3000] + ("..." if len(report_text) > 3000 else ""))

    st.markdown("---")

    # --- Run Analysis ---
    if st.button("Analyze Report for Greenwashing", type="primary", use_container_width=True):

        if use_fallback:
            with st.spinner("Running rule-based analysis..."):
                result = _fallback_analysis(report_text)
        else:
            with st.spinner("Gemini AI is analyzing the report... (this takes 10-20 seconds)"):
                prompt = _build_analysis_prompt(report_text, company_match)
                raw_response = _call_gemini(api_key, prompt)

                if raw_response.startswith("ERROR:"):
                    st.error(f"Gemini API error: {raw_response}")
                    st.warning("Falling back to rule-based analysis...")
                    result = _fallback_analysis(report_text)
                else:
                    result = _parse_gemini_response(raw_response)
                    if result is None:
                        st.warning("Could not parse Gemini response. Using rule-based fallback...")
                        result = _fallback_analysis(report_text)

        # Store result in session state for persistence
        st.session_state['report_analysis'] = result
        st.session_state['report_name'] = uploaded_file.name

    # --- Display Results ---
    if 'report_analysis' not in st.session_state:
        return

    result = st.session_state['report_analysis']
    report_name = st.session_state.get('report_name', 'Report')

    st.markdown("---")
    st.header(f"Analysis Results: {report_name}")

    # === SECTION 1: Overall Risk Score ===
    col1, col2, col3, col4 = st.columns(4)
    overall_score = result.get('overall_score', 50)
    overall_risk = result.get('overall_risk', 'MEDIUM')
    claims = result.get('claims', [])

    with col1:
        st.metric("Greenwashing Risk", f"{overall_score}/100")
    with col2:
        st.metric("Risk Level", overall_risk)
    with col3:
        st.metric("Claims Analyzed", len(claims))
    with col4:
        high_claims = sum(1 for c in claims if c.get('risk_level') == 'HIGH')
        st.metric("Suspicious Claims", f"{high_claims}/{len(claims)}")

    # Overall verdict
    if overall_risk == 'HIGH':
        st.error(f"**HIGH GREENWASHING RISK** -- {result.get('summary', '')}")
    elif overall_risk == 'MEDIUM':
        st.warning(f"**MODERATE GREENWASHING RISK** -- {result.get('summary', '')}")
    else:
        st.success(f"**LOW GREENWASHING RISK** -- {result.get('summary', '')}")

    # Risk gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall_score,
        title={'text': "Greenwashing Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#e74c3c' if overall_score >= 60 else '#ff9800' if overall_score >= 40 else '#4caf50'},
            'steps': [
                {'range': [0, 30], 'color': '#e8f5e9'},
                {'range': [30, 60], 'color': '#fff3e0'},
                {'range': [60, 100], 'color': '#ffebee'},
            ],
        }
    ))
    fig.update_layout(height=280)
    st.plotly_chart(fig, use_container_width=True)

    # === Cross-check with existing data ===
    if company_match:
        st.markdown("---")
        st.subheader("Cross-Check with Existing Data")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Our ML Risk Score",
                      f"{company_match.get('risk_score', 'N/A'):.1f}/100")
        with col2:
            st.metric("Report Risk Score", f"{overall_score}/100")
        with col3:
            diff = overall_score - float(company_match.get('risk_score', 50))
            st.metric("Discrepancy", f"{diff:+.1f}",
                      delta=f"{'Report scores higher' if diff > 0 else 'Report scores lower'}",
                      delta_color="inverse")
        if abs(diff) > 20:
            st.error(f"Significant discrepancy between report analysis ({overall_score}) "
                     f"and ML model ({company_match.get('risk_score', 'N/A'):.1f}). "
                     f"The report may be presenting a skewed picture.")

    st.markdown("---")

    # === SECTION 2: Claim-by-Claim Analysis ===
    st.subheader("Claim-by-Claim Analysis")
    st.markdown("Each ESG claim is extracted, categorized, and assessed for greenwashing risk:")

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        filter_risk = st.multiselect("Filter by risk level",
            ['HIGH', 'MEDIUM', 'LOW'], default=['HIGH', 'MEDIUM', 'LOW'])
    with col2:
        filter_cat = st.multiselect("Filter by category",
            ['Environmental', 'Social', 'Governance'],
            default=['Environmental', 'Social', 'Governance'])

    for i, claim in enumerate(claims):
        risk = claim.get('risk_level', 'MEDIUM')
        category = claim.get('category', 'Unknown')

        if risk not in filter_risk or category not in filter_cat:
            continue

        css_class = f"claim-{risk.lower()}"
        label_class = f"label-{risk.lower()}"
        icon = '🔴' if risk == 'HIGH' else '🟡' if risk == 'MEDIUM' else '🟢'
        evidence = claim.get('evidence_type', 'Unknown')

        st.markdown(
            f'<div class="{css_class}">'
            f'<span class="claim-label {label_class}">{risk}</span>'
            f'<span class="claim-label" style="background:#3498db;">{category}</span>'
            f'<span class="claim-label" style="background:#8e44ad;">{evidence}</span>'
            f'<br><br>'
            f'{icon} <b>Claim:</b> "<i>{claim.get("claim_text", "")}</i>"'
            f'<br><br>'
            f'<b>Issue:</b> {claim.get("issue", "")}'
            f'<br>'
            f'<b>Explanation:</b> {claim.get("explanation", "")}'
            f'</div>',
            unsafe_allow_html=True)

    st.markdown("---")

    # === SECTION 3: Risk Distribution Chart ===
    st.subheader("Claims Risk Distribution")
    col1, col2 = st.columns(2)

    with col1:
        risk_counts = pd.DataFrame([
            {'Risk Level': c.get('risk_level', 'MEDIUM')} for c in claims
        ])['Risk Level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        fig = px.pie(risk_counts, values='Count', names='Risk Level',
                     color='Risk Level',
                     color_discrete_map={'HIGH': '#e74c3c', 'MEDIUM': '#ff9800', 'LOW': '#4caf50'},
                     title='Claims by Risk Level', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cat_counts = pd.DataFrame([
            {'Category': c.get('category', 'Unknown')} for c in claims
        ])['Category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        fig = px.bar(cat_counts, x='Category', y='Count',
                     color='Category',
                     color_discrete_map={'Environmental': '#2ecc71', 'Social': '#3498db',
                                         'Governance': '#9b59b6'},
                     title='Claims by ESG Category')
        st.plotly_chart(fig, use_container_width=True)

    # Evidence type breakdown
    evidence_counts = pd.DataFrame([
        {'Evidence': c.get('evidence_type', 'Unknown')} for c in claims
    ])['Evidence'].value_counts().reset_index()
    evidence_counts.columns = ['Evidence Type', 'Count']
    fig = px.bar(evidence_counts, x='Evidence Type', y='Count',
                 color='Evidence Type',
                 color_discrete_map={
                     'Vague Promise': '#e74c3c', 'Missing Data': '#ff9800',
                     'Contradicts Data': '#c0392b', 'Unverified': '#f39c12',
                     'Credible': '#27ae60',
                 },
                 title='Evidence Type Distribution')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # === SECTION 4: Red Flags & Positive Signals ===
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Red Flags")
        red_flags = result.get('red_flags', [])
        if red_flags:
            for flag in red_flags:
                st.markdown(f'<div class="red-flag-box">⚠ {flag}</div>',
                            unsafe_allow_html=True)
        else:
            st.success("No major red flags detected.")

    with col2:
        st.subheader("Positive Signals")
        positives = result.get('positive_signals', [])
        if positives:
            for pos in positives:
                st.markdown(f'<div class="positive-box">✓ {pos}</div>',
                            unsafe_allow_html=True)
        else:
            st.info("No strong positive signals found.")

    st.markdown("---")

    # === SECTION 5: Recommendations ===
    st.subheader("Recommendations")
    recommendations = result.get('recommendations', [])
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"**{i}.** {rec}")

    st.markdown("---")

    # === SECTION 6: Export Results ===
    st.subheader("Export Analysis")
    export_data = json.dumps(result, indent=2, ensure_ascii=False)
    st.download_button(
        label="Download Full Analysis (JSON)",
        data=export_data,
        file_name=f"esg_analysis_{report_name.replace('.pdf', '')}.json",
        mime="application/json",
    )

    # CSV export of claims
    if claims:
        claims_df = pd.DataFrame(claims)
        csv_data = claims_df.to_csv(index=False)
        st.download_button(
            label="Download Claims Table (CSV)",
            data=csv_data,
            file_name=f"esg_claims_{report_name.replace('.pdf', '')}.csv",
            mime="text/csv",
        )


# ============================================================================
# PAGE 8: REPORT GENERATOR
# ============================================================================

def _generate_html_report(company_name, risk_data, fm_data, pred_data, fi_df, all_risk_df):
    """Generate a full HTML greenwashing analysis report for a company."""

    risk_score = float(risk_data.get('risk_score', 0))
    risk_tier = risk_data.get('risk_tier', 'N/A')
    sector = risk_data.get('sector', 'N/A')
    esg_risk = float(risk_data.get('total_esg_risk_score', 0))
    controversy = float(risk_data.get('controversy_score', 0))

    # Component scores
    comp_proxy = float(risk_data.get('comp_proxy', 0))
    comp_linguistic = float(risk_data.get('comp_linguistic', 0))
    comp_divergence = float(risk_data.get('comp_divergence', 0))
    comp_credibility = float(risk_data.get('comp_credibility_inv', 0))
    comp_controversy = float(risk_data.get('comp_controversy_ratio', 0))

    # Feature matrix data
    env_risk = float(fm_data.get('env_risk_score', 0)) if fm_data is not None else 0
    social_risk = float(fm_data.get('social_risk_score', 0)) if fm_data is not None else 0
    gov_risk = float(fm_data.get('gov_risk_score', 0)) if fm_data is not None else 0
    gw_signal = float(fm_data.get('greenwashing_signal_score', 0)) if fm_data is not None else 0
    vague_count = int(fm_data.get('vague_language_count', 0)) if fm_data is not None else 0
    concrete_count = int(fm_data.get('concrete_evidence_count', 0)) if fm_data is not None else 0
    hedge_count = int(fm_data.get('hedge_language_count', 0)) if fm_data is not None else 0
    superlative_count = int(fm_data.get('superlative_count', 0)) if fm_data is not None else 0
    future_count = int(fm_data.get('future_language_count', 0)) if fm_data is not None else 0
    sentiment = float(fm_data.get('text_polarity', 0)) if fm_data is not None else 0
    flesch = float(fm_data.get('flesch_reading_ease', 0)) if fm_data is not None else 0
    divergence = float(fm_data.get('esg_controversy_divergence', 0)) if fm_data is not None else 0
    controversy_ratio = float(fm_data.get('controversy_risk_ratio', 0)) if fm_data is not None else 0
    imbalance = float(fm_data.get('pillar_imbalance_score', 0)) if fm_data is not None else 0
    anomaly = float(fm_data.get('combined_anomaly_score', 0)) if fm_data is not None else 0
    mismatch = int(fm_data.get('risk_controversy_mismatch', 0)) if fm_data is not None else 0
    lexical_div = float(fm_data.get('lexical_diversity', 0)) if fm_data is not None else 0

    # Enhanced NLP features (Categories 7-10)
    aggregate_nlp = float(fm_data.get('aggregate_esg_nlp_score', 50)) if fm_data is not None else 50
    reg_readiness = float(fm_data.get('regulatory_readiness_score', 0)) if fm_data is not None else 0
    policy_gap = float(fm_data.get('policy_esg_gap', 0.5)) if fm_data is not None else 0.5
    narrative_cred = float(fm_data.get('narrative_credibility_index', 0.5)) if fm_data is not None else 0.5
    commit_cred = float(fm_data.get('commitment_credibility_score', 0.5)) if fm_data is not None else 0.5
    multi_gw = float(fm_data.get('multi_signal_greenwashing_score', 0.5)) if fm_data is not None else 0.5
    temporal_gw = float(fm_data.get('temporal_greenwashing_signal', 0.5)) if fm_data is not None else 0.5
    news_gw = float(fm_data.get('news_greenwashing_signal', 0.5)) if fm_data is not None else 0.5

    # Predictions
    proxy_score = int(pred_data.get('gw_proxy_score', 0)) if pred_data is not None else 0
    label = int(pred_data.get('gw_label_binary', 0)) if pred_data is not None else 0

    # Sector peers
    peer_df = all_risk_df[all_risk_df['sector'] == sector] if sector != 'N/A' else pd.DataFrame()
    peer_count = len(peer_df)
    sector_avg = float(peer_df['risk_score'].mean()) if not peer_df.empty else 0
    peer_rank = int((peer_df['risk_score'] > risk_score).sum()) + 1 if not peer_df.empty else 0

    # Risk color
    if risk_score >= 60:
        risk_color = '#e74c3c'
        verdict = 'HIGH GREENWASHING RISK'
        verdict_detail = 'Multiple indicators suggest this company may be engaging in greenwashing practices.'
    elif risk_score >= 40:
        risk_color = '#ff9800'
        verdict = 'MODERATE GREENWASHING RISK'
        verdict_detail = 'Some indicators warrant further investigation into ESG claims.'
    else:
        risk_color = '#27ae60'
        verdict = 'LOW GREENWASHING RISK'
        verdict_detail = 'ESG claims appear largely consistent with actual performance metrics.'

    # Indicator rows
    def indicator_row(name, value, threshold, desc):
        status = 'HIGH' if value > threshold else 'Normal'
        s_color = '#e74c3c' if status == 'HIGH' else '#27ae60'
        return f"""<tr>
            <td>{name}</td><td>{value:.4f}</td><td>{threshold:.4f}</td>
            <td style="color:{s_color};font-weight:bold;">{status}</td>
            <td style="font-size:12px;color:#666;">{desc}</td></tr>"""

    now = datetime.now().strftime('%B %d, %Y at %H:%M')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ESG Greenwashing Report -- {company_name}</title>
<style>
  @page {{ margin: 30px; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #333; margin: 0; padding: 0; background: #fff; }}
  .cover {{ background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460); color: white;
            padding: 60px 50px; text-align: center; }}
  .cover h1 {{ font-size: 36px; margin: 0 0 10px 0; color: #e94560; }}
  .cover h2 {{ font-size: 24px; margin: 0 0 20px 0; font-weight: normal; }}
  .cover .meta {{ font-size: 14px; color: #7ec8e3; }}
  .content {{ padding: 40px 50px; }}
  h2 {{ color: #0f3460; border-bottom: 3px solid #e94560; padding-bottom: 8px; margin-top: 40px; }}
  h3 {{ color: #16213e; margin-top: 25px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
  .kpi-card {{ background: #f8f9fa; border-radius: 10px; padding: 20px; text-align: center;
               border-top: 4px solid #3498db; }}
  .kpi-card .value {{ font-size: 28px; font-weight: bold; color: #0f3460; }}
  .kpi-card .label {{ font-size: 13px; color: #666; margin-top: 5px; }}
  .verdict {{ background: {risk_color}; color: white; padding: 20px 30px; border-radius: 10px;
              font-size: 18px; text-align: center; margin: 20px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 14px; }}
  th {{ background: #0f3460; color: white; padding: 10px 12px; text-align: left; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #eee; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  .component-bar {{ height: 22px; border-radius: 4px; display: inline-block; min-width: 30px;
                    color: white; font-size: 12px; text-align: center; line-height: 22px; }}
  .bar-container {{ background: #eee; border-radius: 4px; width: 100%; position: relative; }}
  .section-box {{ background: #f0f4ff; border-left: 4px solid #3498db; padding: 15px 20px;
                  border-radius: 0 8px 8px 0; margin: 15px 0; }}
  .red-box {{ background: #fff0f0; border-left: 4px solid #e74c3c; padding: 12px 18px;
              border-radius: 0 8px 8px 0; margin: 8px 0; }}
  .green-box {{ background: #f0fff0; border-left: 4px solid #27ae60; padding: 12px 18px;
                border-radius: 0 8px 8px 0; margin: 8px 0; }}
  .yellow-box {{ background: #fffde7; border-left: 4px solid #ff9800; padding: 12px 18px;
                 border-radius: 0 8px 8px 0; margin: 8px 0; }}
  .footer {{ background: #1a1a2e; color: #7ec8e3; padding: 20px 50px; text-align: center;
             font-size: 12px; margin-top: 40px; }}
  .gauge-container {{ text-align: center; margin: 20px auto; }}
  .gauge {{ width: 200px; height: 100px; border-radius: 200px 200px 0 0;
            background: conic-gradient(#27ae60 0% 20%, #82e0aa 20% 40%, #f4d03f 40% 60%,
            #e67e22 60% 80%, #e74c3c 80% 100%);
            position: relative; display: inline-block; overflow: hidden; }}
  .pillar-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }}
  .pillar-card {{ text-align: center; padding: 20px; border-radius: 10px; }}
  .pillar-card .score {{ font-size: 32px; font-weight: bold; }}
  .env-card {{ background: #e8f5e9; border: 2px solid #4caf50; }}
  .env-card .score {{ color: #2e7d32; }}
  .social-card {{ background: #e3f2fd; border: 2px solid #2196f3; }}
  .social-card .score {{ color: #1565c0; }}
  .gov-card {{ background: #f3e5f5; border: 2px solid #9c27b0; }}
  .gov-card .score {{ color: #6a1b9a; }}
</style>
</head>
<body>

<!-- COVER PAGE -->
<div class="cover">
  <h1>ESG GREENWASHING ANALYSIS REPORT</h1>
  <h2>{company_name}</h2>
  <div class="meta">
    Sector: {sector} | Generated: {now}<br>
    ESG Greenwashing Detection System | NLP + Machine Learning + Explainable AI
  </div>
</div>

<div class="content">

<!-- EXECUTIVE SUMMARY -->
<h2>1. Executive Summary</h2>

<div class="verdict">{verdict}: {company_name} scores {risk_score:.1f}/100 on greenwashing risk</div>

<p>{verdict_detail}</p>

<div class="kpi-grid">
  <div class="kpi-card" style="border-top-color:{risk_color};">
    <div class="value" style="color:{risk_color};">{risk_score:.1f}</div>
    <div class="label">Risk Score (0-100)</div>
  </div>
  <div class="kpi-card">
    <div class="value">{risk_tier}</div>
    <div class="label">Risk Tier</div>
  </div>
  <div class="kpi-card">
    <div class="value">{esg_risk:.1f}</div>
    <div class="label">Total ESG Risk</div>
  </div>
  <div class="kpi-card">
    <div class="value">{controversy:.1f}</div>
    <div class="label">Controversy Score</div>
  </div>
</div>

<div class="section-box">
  <strong>Model Prediction:</strong> This company triggered <strong>{proxy_score} out of 5</strong>
  greenwashing indicators and is classified as
  <strong>{"FLAGGED -- Potential Greenwashing" if label == 1 else "NOT FLAGGED -- Low Risk"}</strong>
  by the Gradient Boosting model (F1 = 0.9682).
</div>

<!-- RISK SCORE BREAKDOWN -->
<h2>2. Risk Score Breakdown</h2>

<p>The greenwashing risk score is a weighted composite of 5 components:</p>

<table>
  <tr><th>Component</th><th>Weight</th><th>Score (0-100)</th><th>Visual</th></tr>
  <tr><td>Proxy Score (ML Indicators)</td><td>40%</td><td>{comp_proxy:.1f}</td>
      <td><div class="bar-container"><div class="component-bar" style="width:{comp_proxy}%;background:#e74c3c;">{comp_proxy:.0f}</div></div></td></tr>
  <tr><td>Linguistic GW Signal (NLP)</td><td>15%</td><td>{comp_linguistic:.1f}</td>
      <td><div class="bar-container"><div class="component-bar" style="width:{comp_linguistic}%;background:#ff9800;">{comp_linguistic:.0f}</div></div></td></tr>
  <tr><td>ESG-Controversy Divergence</td><td>15%</td><td>{comp_divergence:.1f}</td>
      <td><div class="bar-container"><div class="component-bar" style="width:{comp_divergence}%;background:#f39c12;">{comp_divergence:.0f}</div></div></td></tr>
  <tr><td>Low Claim Credibility</td><td>15%</td><td>{comp_credibility:.1f}</td>
      <td><div class="bar-container"><div class="component-bar" style="width:{comp_credibility}%;background:#9b59b6;">{comp_credibility:.0f}</div></div></td></tr>
  <tr><td>Controversy-Risk Ratio</td><td>15%</td><td>{comp_controversy:.1f}</td>
      <td><div class="bar-container"><div class="component-bar" style="width:{comp_controversy}%;background:#e67e22;">{comp_controversy:.0f}</div></div></td></tr>
</table>

<!-- ESG PILLAR ANALYSIS -->
<h2>3. ESG Pillar Analysis</h2>

<div class="pillar-grid">
  <div class="pillar-card env-card">
    <div class="score">{env_risk:.1f}</div>
    <div>Environmental Risk</div>
  </div>
  <div class="pillar-card social-card">
    <div class="score">{social_risk:.1f}</div>
    <div>Social Risk</div>
  </div>
  <div class="pillar-card gov-card">
    <div class="score">{gov_risk:.1f}</div>
    <div>Governance Risk</div>
  </div>
</div>

<table>
  <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
  <tr><td>Pillar Imbalance Score</td><td>{imbalance:.4f}</td>
      <td>{"High imbalance -- uneven ESG performance across pillars" if imbalance > 3 else "Balanced ESG profile"}</td></tr>
  <tr><td>Controversy Score</td><td>{controversy:.1f}</td>
      <td>{"High controversy level" if controversy >= 3 else "Moderate controversy" if controversy >= 2 else "Low controversy"}</td></tr>
  <tr><td>ESG-Controversy Divergence</td><td>{divergence:.4f}</td>
      <td>{"CRITICAL: Claims low risk but has high controversy" if divergence > 1 else "Moderate gap" if divergence > 0 else "Consistent profile"}</td></tr>
  <tr><td>Risk-Controversy Mismatch</td><td>{"YES" if mismatch else "NO"}</td>
      <td>{"Red flag: ESG risk level inconsistent with controversy" if mismatch else "ESG risk aligns with controversy level"}</td></tr>
</table>

<!-- NLP LINGUISTIC ANALYSIS -->
<h2>4. NLP & Linguistic Analysis</h2>

<p>Analysis of the company's corporate description text using Natural Language Processing:</p>

<div class="kpi-grid">
  <div class="kpi-card" style="border-top-color:{'#e74c3c' if gw_signal > 0.5 else '#ff9800' if gw_signal > 0.3 else '#27ae60'};">
    <div class="value">{gw_signal:.3f}</div>
    <div class="label">GW Signal Score (0-1)</div>
  </div>
  <div class="kpi-card" style="border-top-color:#e74c3c;">
    <div class="value">{vague_count}</div>
    <div class="label">Vague Language</div>
  </div>
  <div class="kpi-card" style="border-top-color:#27ae60;">
    <div class="value">{concrete_count}</div>
    <div class="label">Concrete Evidence</div>
  </div>
  <div class="kpi-card">
    <div class="value">{sentiment:+.3f}</div>
    <div class="label">Sentiment Polarity</div>
  </div>
</div>

<table>
  <tr><th>Linguistic Feature</th><th>Count</th><th>Risk Implication</th></tr>
  <tr><td>Vague Language ("committed to", "striving for")</td><td>{vague_count}</td>
      <td style="color:{'#e74c3c' if vague_count > 3 else '#333'};">{"High -- excessive vague promises" if vague_count > 3 else "Acceptable"}</td></tr>
  <tr><td>Hedge Language ("approximately", "potentially")</td><td>{hedge_count}</td>
      <td style="color:{'#e74c3c' if hedge_count > 3 else '#333'};">{"High -- excessive hedging" if hedge_count > 3 else "Acceptable"}</td></tr>
  <tr><td>Superlatives ("industry-leading", "world-class")</td><td>{superlative_count}</td>
      <td style="color:{'#ff9800' if superlative_count > 2 else '#333'};">{"Elevated -- unsubstantiated claims" if superlative_count > 2 else "Normal"}</td></tr>
  <tr><td>Future Language ("will", "plan to", "by 2030")</td><td>{future_count}</td>
      <td style="color:{'#ff9800' if future_count > 3 else '#333'};">{"Heavy future focus without past data" if future_count > 3 else "Balanced"}</td></tr>
  <tr><td>Concrete Evidence ("reduced by X%", "ISO certified")</td><td>{concrete_count}</td>
      <td style="color:{'#27ae60' if concrete_count > 2 else '#e74c3c'};">{"Good -- includes verifiable data" if concrete_count > 2 else "Low -- lacks measurable evidence"}</td></tr>
  <tr><td>Flesch Reading Ease</td><td>{flesch:.1f}</td>
      <td>{"Easy to read" if flesch > 60 else "Moderate complexity" if flesch > 30 else "Very complex -- potential obfuscation"}</td></tr>
  <tr><td>Lexical Diversity</td><td>{lexical_div:.3f}</td>
      <td>{"High vocabulary diversity" if lexical_div > 0.6 else "Repetitive language"}</td></tr>
</table>

{"<div class='red-box'><strong>LINGUISTIC RED FLAG:</strong> Vague language count (" + str(vague_count) + ") significantly exceeds concrete evidence (" + str(concrete_count) + "). This is a classic greenwashing pattern.</div>" if vague_count > concrete_count * 1.5 and vague_count > 2 else ""}
{"<div class='green-box'><strong>POSITIVE:</strong> Concrete evidence (" + str(concrete_count) + ") outweighs vague language (" + str(vague_count) + "). Corporate text appears substantive.</div>" if concrete_count >= vague_count and concrete_count > 1 else ""}

<h3>4b. Enhanced ESG Intelligence (Policy, News, Temporal)</h3>

<div class="kpi-grid">
  <div class="kpi-card" style="border-top-color:{'#27ae60' if aggregate_nlp > 60 else '#ff9800' if aggregate_nlp > 35 else '#e74c3c'};">
    <div class="value">{aggregate_nlp:.1f}/100</div>
    <div class="label">Aggregate ESG NLP Score</div>
  </div>
  <div class="kpi-card" style="border-top-color:{'#27ae60' if reg_readiness > 0.4 else '#ff9800' if reg_readiness > 0.15 else '#e74c3c'};">
    <div class="value">{reg_readiness:.3f}</div>
    <div class="label">Regulatory Readiness</div>
  </div>
  <div class="kpi-card" style="border-top-color:{'#27ae60' if narrative_cred > 0.5 else '#ff9800' if narrative_cred > 0.3 else '#e74c3c'};">
    <div class="value">{narrative_cred:.3f}</div>
    <div class="label">Narrative Credibility</div>
  </div>
  <div class="kpi-card" style="border-top-color:{'#27ae60' if commit_cred > 0.6 else '#ff9800' if commit_cred > 0.3 else '#e74c3c'};">
    <div class="value">{commit_cred:.3f}</div>
    <div class="label">Commitment Credibility</div>
  </div>
</div>

<table>
  <tr><th>Enhanced Metric</th><th>Value</th><th>Risk Implication</th></tr>
  <tr><td>Policy-ESG Gap</td><td>{policy_gap:.3f}</td>
      <td style="color:{'#e74c3c' if policy_gap > 0.6 else '#333'};">{"High -- ESG talk without policy grounding" if policy_gap > 0.6 else "Acceptable"}</td></tr>
  <tr><td>Multi-Signal GW Score</td><td>{multi_gw:.3f}</td>
      <td style="color:{'#e74c3c' if multi_gw > 0.6 else '#ff9800' if multi_gw > 0.4 else '#333'};">{"High -- multiple GW signals" if multi_gw > 0.6 else "Moderate" if multi_gw > 0.4 else "Low risk"}</td></tr>
  <tr><td>News GW Signal</td><td>{news_gw:.3f}</td>
      <td style="color:{'#e74c3c' if news_gw > 0.5 else '#333'};">{"Promotional narrative dominates" if news_gw > 0.5 else "Balanced narrative"}</td></tr>
  <tr><td>Temporal GW Signal</td><td>{temporal_gw:.3f}</td>
      <td style="color:{'#e74c3c' if temporal_gw > 0.6 else '#333'};">{"Promises exceed past delivery" if temporal_gw > 0.6 else "Credible temporal pattern"}</td></tr>
</table>

{"<div class='red-box'><strong>POLICY GAP ALERT:</strong> Company uses ESG language without referencing recognized regulatory frameworks. Regulatory readiness = " + f'{reg_readiness:.3f}' + ".</div>" if reg_readiness < 0.15 and policy_gap > 0.6 else ""}
{"<div class='green-box'><strong>STRONG POLICY ALIGNMENT:</strong> Company references recognized ESG frameworks with regulatory readiness = " + f'{reg_readiness:.3f}' + ".</div>" if reg_readiness > 0.4 else ""}

<!-- KEY GREENWASHING INDICATORS -->
<h2>5. Key Greenwashing Indicators</h2>

<p>Six critical indicators are evaluated against the 75th percentile threshold of the 480-company population:</p>

<table>
  <tr><th>Indicator</th><th>Value</th><th>Threshold (75th pct)</th><th>Status</th><th>Description</th></tr>
  {indicator_row('Controversy-Risk Ratio', controversy_ratio, 0.15, 'High controversy relative to ESG risk')}
  {indicator_row('ESG-Controversy Divergence', divergence, 0.5, 'Gap between ESG claims and actual controversy')}
  {indicator_row('GW Linguistic Score', gw_signal, 0.45, 'NLP-detected greenwashing language patterns')}
  {indicator_row('Pillar Imbalance', imbalance, 3.5, 'Uneven ESG performance across E/S/G')}
  {indicator_row('Combined Anomaly Score', anomaly, 1.5, 'Statistical anomaly across multiple features')}
</table>

<!-- SECTOR COMPARISON -->
<h2>6. Sector Peer Comparison</h2>

<div class="section-box">
  <strong>{company_name}</strong> ranks <strong>#{peer_rank} out of {peer_count}</strong>
  in the <strong>{sector}</strong> sector by greenwashing risk score.<br>
  Company score: <strong>{risk_score:.1f}</strong> | Sector average: <strong>{sector_avg:.1f}</strong>
  | Difference: <strong>{risk_score - sector_avg:+.1f}</strong>
</div>

{"<div class='red-box'>This company scores <strong>" + f"{risk_score - sector_avg:.1f}" + " points above</strong> its sector average, indicating higher greenwashing risk relative to peers.</div>" if risk_score > sector_avg + 10 else ""}
{"<div class='green-box'>This company scores <strong>" + f"{sector_avg - risk_score:.1f}" + " points below</strong> its sector average, indicating lower greenwashing risk relative to peers.</div>" if risk_score < sector_avg - 10 else ""}

<!-- METHODOLOGY -->
<h2>7. Methodology</h2>

<div class="section-box">
<strong>Data Sources:</strong> S&P 500 ESG Risk Ratings, NIFTY 50 ESG Data, ESG Financial Dataset, Greenwashing Score Data<br>
<strong>Companies Analyzed:</strong> 480 (430 S&P 500 + 50 NIFTY 50)<br>
<strong>Features Engineered:</strong> 161 (36 Numerical + 47 NLP + 31 Categorical + Scaled Variants)<br>
<strong>Best Model:</strong> Gradient Boosting (F1 = 0.9682, ROC-AUC = 0.9979)<br>
<strong>Explainability:</strong> SHAP (SHapley Additive exPlanations) for feature attribution<br>
<strong>Risk Score:</strong> Weighted composite: 40% Proxy + 15% Linguistic + 15% Divergence + 15% Credibility + 15% Controversy Ratio
</div>

<h3>7-Phase Pipeline</h3>
<table>
  <tr><th>Phase</th><th>Description</th></tr>
  <tr><td>1. Data Collection</td><td>4 datasets from Kaggle (12K+ rows)</td></tr>
  <tr><td>2. Preprocessing</td><td>Cleaning, encoding, merging into 480-company profiles</td></tr>
  <tr><td>3. NLP Text Analysis</td><td>Sentiment, readability, ESG keywords, greenwashing linguistics, claim extraction</td></tr>
  <tr><td>4. Feature Engineering</td><td>161 features: numerical ratios, NLP metrics, categorical encodings</td></tr>
  <tr><td>5. Model Training</td><td>5 supervised models + Isolation Forest with GridSearchCV (5-fold CV)</td></tr>
  <tr><td>6. Evaluation & SHAP</td><td>ROC curves, confusion matrices, SHAP per-company explanations</td></tr>
  <tr><td>7. Risk Scoring</td><td>Composite 0-100 score with 5 weighted components</td></tr>
</table>

</div>

<div class="footer">
  ESG Greenwashing Detection System | VNR VJIET Team-18 | Generated {now}<br>
  Powered by: Python, Scikit-learn, XGBoost, SHAP, NLP, Streamlit, Google Gemini
</div>

</body>
</html>"""

    return html


def page_report_generator(data):
    """Generate downloadable HTML/PDF reports for any company."""

    st.markdown("""
    <style>
    .report-header {
        background: linear-gradient(135deg, #0f3460, #16213e, #1a1a2e);
        padding: 25px; border-radius: 12px; margin-bottom: 20px;
        border: 1px solid #3498db;
    }
    .report-header h1 { color: #3498db; margin: 0; font-family: monospace; }
    .report-header p { color: #7ec8e3; margin: 5px 0 0 0; font-size: 14px; }
    </style>
    <div class="report-header">
        <h1>ESG GREENWASHING REPORT GENERATOR</h1>
        <p>Generate professional, downloadable analysis reports for any company</p>
    </div>
    """, unsafe_allow_html=True)

    df = data['risk_scores']
    fm = data['feature_matrix']
    pred = data['predictions']
    fi = data['feature_importance']

    if df.empty:
        st.warning("Data not available. Run `python model_pipeline.py` first.")
        return

    # --- Report configuration ---
    st.subheader("Configure Report")

    col1, col2 = st.columns(2)
    with col1:
        company_names = sorted(df['company_name'].dropna().unique().tolist())
        mode = st.radio("Report mode", ["Single Company", "Batch (Multiple)", "Full Portfolio"])

    if mode == "Single Company":
        with col2:
            selected = st.selectbox("Select company", company_names)

        if selected and st.button("Generate Report", type="primary", use_container_width=True):
            with st.spinner(f"Generating report for {selected}..."):
                risk_row = df[df['company_name'] == selected].iloc[0].to_dict()
                fm_row = fm[fm['company_name'] == selected].iloc[0].to_dict() if not fm.empty and selected in fm['company_name'].values else None
                pred_row = pred[pred['company_name'] == selected].iloc[0].to_dict() if not pred.empty and selected in pred['company_name'].values else None

                html = _generate_html_report(selected, risk_row, fm_row, pred_row, fi, df)

            st.success(f"Report generated for **{selected}**!")

            # Preview
            st.subheader("Report Preview")
            st.components.v1.html(html, height=800, scrolling=True)

            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download HTML Report",
                    data=html,
                    file_name=f"ESG_Report_{selected.replace(' ', '_')}.html",
                    mime="text/html",
                    use_container_width=True,
                )
            with col2:
                # CSV summary
                summary = {
                    'Company': selected,
                    'Sector': risk_row.get('sector', ''),
                    'Risk Score': risk_row.get('risk_score', ''),
                    'Risk Tier': risk_row.get('risk_tier', ''),
                    'ESG Risk': risk_row.get('total_esg_risk_score', ''),
                    'Controversy': risk_row.get('controversy_score', ''),
                    'GW Signal': fm_row.get('greenwashing_signal_score', '') if fm_row else '',
                    'Proxy Score': pred_row.get('gw_proxy_score', '') if pred_row else '',
                    'Label': 'Flagged' if pred_row and pred_row.get('gw_label_binary') == 1 else 'Not Flagged',
                }
                csv = pd.DataFrame([summary]).to_csv(index=False)
                st.download_button(
                    label="Download Summary CSV",
                    data=csv,
                    file_name=f"ESG_Summary_{selected.replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    elif mode == "Batch (Multiple)":
        with col2:
            selected_companies = st.multiselect("Select companies", company_names,
                                                default=company_names[:3])

        if selected_companies and st.button("Generate Batch Reports", type="primary",
                                             use_container_width=True):
            progress = st.progress(0)
            all_summaries = []

            for i, comp in enumerate(selected_companies):
                progress.progress((i + 1) / len(selected_companies))
                risk_row = df[df['company_name'] == comp].iloc[0].to_dict()
                fm_row = fm[fm['company_name'] == comp].iloc[0].to_dict() if not fm.empty and comp in fm['company_name'].values else None
                pred_row = pred[pred['company_name'] == comp].iloc[0].to_dict() if not pred.empty and comp in pred['company_name'].values else None

                html = _generate_html_report(comp, risk_row, fm_row, pred_row, fi, df)

                st.download_button(
                    label=f"Download: {comp}",
                    data=html,
                    file_name=f"ESG_Report_{comp.replace(' ', '_')}.html",
                    mime="text/html",
                    key=f"dl_{comp}",
                )

                all_summaries.append({
                    'Company': comp,
                    'Sector': risk_row.get('sector', ''),
                    'Risk Score': risk_row.get('risk_score', ''),
                    'Risk Tier': risk_row.get('risk_tier', ''),
                    'ESG Risk': risk_row.get('total_esg_risk_score', ''),
                    'Controversy': risk_row.get('controversy_score', ''),
                })

            progress.empty()
            st.success(f"Generated {len(selected_companies)} reports!")

            # Batch summary table
            batch_df = pd.DataFrame(all_summaries)
            st.dataframe(batch_df, use_container_width=True)

            batch_csv = batch_df.to_csv(index=False)
            st.download_button(
                label="Download Batch Summary CSV",
                data=batch_csv,
                file_name="ESG_Batch_Summary.csv",
                mime="text/csv",
            )

    elif mode == "Full Portfolio":
        st.markdown(f"Generate summary report for all **{len(df)}** companies.")

        if st.button("Generate Portfolio Report", type="primary", use_container_width=True):
            with st.spinner("Generating portfolio report..."):
                # Portfolio summary stats
                total = len(df)
                high_risk = len(df[df['risk_score'] >= 60])
                moderate = len(df[(df['risk_score'] >= 40) & (df['risk_score'] < 60)])
                low_risk = len(df[df['risk_score'] < 40])

                sector_summary = df.groupby('sector').agg(
                    Companies=('risk_score', 'count'),
                    Avg_Risk=('risk_score', 'mean'),
                    Max_Risk=('risk_score', 'max'),
                    High_Risk_Count=('risk_score', lambda x: (x >= 60).sum()),
                ).round(1).sort_values('Avg_Risk', ascending=False)

                sector_rows = ""
                for sec, row in sector_summary.iterrows():
                    color = '#e74c3c' if row['Avg_Risk'] >= 50 else '#ff9800' if row['Avg_Risk'] >= 35 else '#27ae60'
                    sector_rows += f"""<tr><td>{sec}</td><td>{int(row['Companies'])}</td>
                        <td style="color:{color};font-weight:bold;">{row['Avg_Risk']:.1f}</td>
                        <td>{row['Max_Risk']:.1f}</td><td>{int(row['High_Risk_Count'])}</td></tr>"""

                top20 = df.nlargest(20, 'risk_score')
                top20_rows = ""
                for _, row in top20.iterrows():
                    top20_rows += f"""<tr><td>{row['company_name']}</td><td>{row['sector']}</td>
                        <td style="color:#e74c3c;font-weight:bold;">{row['risk_score']:.1f}</td>
                        <td>{row.get('risk_tier', '')}</td></tr>"""

                now = datetime.now().strftime('%B %d, %Y at %H:%M')

                portfolio_html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>ESG Portfolio Greenwashing Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #333; margin: 0; }}
  .cover {{ background: linear-gradient(135deg, #1a1a2e, #0f3460); color: white;
            padding: 60px 50px; text-align: center; }}
  .cover h1 {{ font-size: 36px; color: #e94560; margin: 0 0 10px 0; }}
  .cover h2 {{ font-weight: normal; font-size: 20px; }}
  .cover .meta {{ font-size: 14px; color: #7ec8e3; }}
  .content {{ padding: 40px 50px; }}
  h2 {{ color: #0f3460; border-bottom: 3px solid #e94560; padding-bottom: 8px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
  .kpi-card {{ background: #f8f9fa; border-radius: 10px; padding: 20px; text-align: center;
               border-top: 4px solid #3498db; }}
  .kpi-card .value {{ font-size: 32px; font-weight: bold; color: #0f3460; }}
  .kpi-card .label {{ font-size: 13px; color: #666; margin-top: 5px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 14px; }}
  th {{ background: #0f3460; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  .footer {{ background: #1a1a2e; color: #7ec8e3; padding: 20px; text-align: center; font-size: 12px; margin-top: 40px; }}
</style></head><body>

<div class="cover">
  <h1>PORTFOLIO ESG GREENWASHING REPORT</h1>
  <h2>{total} Companies | S&P 500 + NIFTY 50</h2>
  <div class="meta">Generated: {now}</div>
</div>

<div class="content">
<h2>Portfolio Overview</h2>
<div class="kpi-grid">
  <div class="kpi-card"><div class="value">{total}</div><div class="label">Total Companies</div></div>
  <div class="kpi-card" style="border-top-color:#e74c3c;"><div class="value" style="color:#e74c3c;">{high_risk}</div><div class="label">High Risk (60+)</div></div>
  <div class="kpi-card" style="border-top-color:#ff9800;"><div class="value" style="color:#ff9800;">{moderate}</div><div class="label">Moderate (40-60)</div></div>
  <div class="kpi-card" style="border-top-color:#27ae60;"><div class="value" style="color:#27ae60;">{low_risk}</div><div class="label">Low Risk (&lt;40)</div></div>
</div>

<h2>Sector-Level Summary</h2>
<table><tr><th>Sector</th><th>Companies</th><th>Avg Risk</th><th>Max Risk</th><th>High Risk Count</th></tr>
{sector_rows}</table>

<h2>Top 20 Highest Risk Companies</h2>
<table><tr><th>Company</th><th>Sector</th><th>Risk Score</th><th>Risk Tier</th></tr>
{top20_rows}</table>

</div>
<div class="footer">ESG Greenwashing Detection System | VNR VJIET Team-18 | {now}</div>
</body></html>"""

            st.success("Portfolio report generated!")
            st.components.v1.html(portfolio_html, height=800, scrolling=True)

            st.download_button(
                label="Download Portfolio Report (HTML)",
                data=portfolio_html,
                file_name="ESG_Portfolio_Report.html",
                mime="text/html",
                use_container_width=True,
            )

            # Full data CSV
            full_csv = df.to_csv(index=False)
            st.download_button(
                label="Download Full Data (CSV)",
                data=full_csv,
                file_name="ESG_Portfolio_Full_Data.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================================
# PAGE 9: ADVANCED EXPLAINABILITY (Counterfactuals + What-If)
# ============================================================================

@st.cache_resource
def _train_lightweight_model(fm_path, processed_dir):
    """Train a lightweight Gradient Boosting model for live predictions."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    fm = pd.read_csv(fm_path)

    # Construct proxy labels (same logic as model_training.py)
    indicator_cols = []
    for col in ['esg_controversy_divergence', 'greenwashing_signal_score',
                'controversy_risk_ratio', 'combined_anomaly_score']:
        if col in fm.columns:
            threshold = fm[col].quantile(0.75)
            fm[f'_ind_{col}'] = (fm[col] > threshold).astype(int)
            indicator_cols.append(f'_ind_{col}')

    if 'risk_controversy_mismatch' in fm.columns:
        fm['_ind_mismatch'] = fm['risk_controversy_mismatch'].fillna(0).astype(int)
        indicator_cols.append('_ind_mismatch')

    fm['_proxy_score'] = fm[indicator_cols].sum(axis=1)
    fm['_label'] = (fm['_proxy_score'] >= 2).astype(int)

    # Prepare features
    drop_cols = ['symbol', 'company_name', 'sector', 'industry', 'description',
                 'source'] + indicator_cols + ['_proxy_score', '_label']
    feature_cols = [c for c in fm.columns if c not in drop_cols
                    and fm[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    X = fm[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = fm['_label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    )
    model.fit(X_scaled, y)

    return model, scaler, feature_cols, fm


def _compute_risk_score_from_features(row, fm_population):
    """Compute a simplified risk score from feature values."""
    score_components = []

    # Component 1: Proxy-like indicators
    indicators = 0
    if 'esg_controversy_divergence' in row and 'esg_controversy_divergence' in fm_population.columns:
        if row['esg_controversy_divergence'] > fm_population['esg_controversy_divergence'].quantile(0.75):
            indicators += 1
    if 'greenwashing_signal_score' in row and 'greenwashing_signal_score' in fm_population.columns:
        if row['greenwashing_signal_score'] > fm_population['greenwashing_signal_score'].quantile(0.75):
            indicators += 1
    if 'controversy_risk_ratio' in row and 'controversy_risk_ratio' in fm_population.columns:
        if row['controversy_risk_ratio'] > fm_population['controversy_risk_ratio'].quantile(0.75):
            indicators += 1
    if 'combined_anomaly_score' in row and 'combined_anomaly_score' in fm_population.columns:
        if row['combined_anomaly_score'] > fm_population['combined_anomaly_score'].quantile(0.75):
            indicators += 1
    if row.get('risk_controversy_mismatch', 0) > 0:
        indicators += 1

    return (indicators / 5) * 100


def page_advanced_explainability(data):
    """Advanced Explainability: Counterfactuals + What-If Sensitivity Analysis."""

    st.markdown("""
    <style>
    .explain-header {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 25px; border-radius: 12px; margin-bottom: 20px;
    }
    .explain-header h1 { color: white; margin: 0; font-family: monospace; }
    .explain-header p { color: #bde0fe; margin: 5px 0 0 0; font-size: 14px; }
    .counterfactual-card {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        padding: 18px; border-radius: 10px; margin: 8px 0;
        border-left: 5px solid #0284c7;
    }
    .cf-arrow { color: #0284c7; font-size: 20px; font-weight: bold; }
    .cf-result-good {
        background: #f0fdf4; padding: 15px; border-radius: 10px;
        border: 2px solid #22c55e; text-align: center; margin: 10px 0;
    }
    .cf-result-bad {
        background: #fef2f2; padding: 15px; border-radius: 10px;
        border: 2px solid #ef4444; text-align: center; margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-header">
        <h1>ADVANCED EXPLAINABILITY ENGINE</h1>
        <p>Counterfactual Explanations | What-If Sensitivity | Feature Impact Simulation</p>
    </div>
    """, unsafe_allow_html=True)

    fm = data['feature_matrix']
    risk_df = data['risk_scores']
    fi = data['feature_importance']

    if fm.empty or risk_df.empty:
        st.warning("Data not available. Run `python model_pipeline.py` first.")
        return

    # Train lightweight model (cached)
    fm_path = os.path.join(PROCESSED_DIR, "feature_matrix_expanded.csv")
    if not os.path.exists(fm_path):
        fm_path = os.path.join(PROCESSED_DIR, "feature_matrix.csv")
    try:
        model, scaler, feature_cols, fm_full = _train_lightweight_model(fm_path, PROCESSED_DIR)
    except Exception as e:
        st.error(f"Could not train prediction model: {e}")
        return

    # Company selector
    company_names = sorted(risk_df['company_name'].dropna().unique().tolist())
    selected = st.selectbox("Select a company to explain", company_names, key='adv_company')

    if not selected:
        return

    company_risk = risk_df[risk_df['company_name'] == selected].iloc[0]
    company_fm = fm_full[fm_full['company_name'] == selected]
    if company_fm.empty:
        st.error(f"Feature data not found for {selected}")
        return
    company_fm = company_fm.iloc[0]

    risk_score = float(company_risk['risk_score'])
    risk_tier = company_risk.get('risk_tier', 'N/A')

    # Current prediction
    X_current = company_fm[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.reshape(1, -1)
    X_current_scaled = scaler.transform(X_current)
    current_prob = model.predict_proba(X_current_scaled)[0][1]
    current_pred = model.predict(X_current_scaled)[0]

    # Header KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Risk Score", f"{risk_score:.1f}/100")
    with col2:
        st.metric("Risk Tier", str(risk_tier))
    with col3:
        st.metric("GW Probability", f"{current_prob:.1%}")
    with col4:
        st.metric("Prediction", "FLAGGED" if current_pred == 1 else "NOT FLAGGED")

    st.markdown("---")

    # === TAB LAYOUT ===
    tab1, tab2, tab3 = st.tabs([
        "Counterfactual Explanations",
        "What-If Sensitivity Sliders",
        "Feature Sensitivity Analysis",
    ])

    # Top features for analysis
    if not fi.empty:
        top_features = [f for f in fi['feature'].head(20).tolist() if f in feature_cols]
    else:
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-20:][::-1]
        top_features = [feature_cols[i] for i in top_idx]

    # ================================================================
    # TAB 1: COUNTERFACTUAL EXPLANATIONS
    # ================================================================
    with tab1:
        st.subheader("Counterfactual Explanations")
        st.markdown(
            "**What minimal changes would flip this company's prediction?**  \n"
            "Each counterfactual shows the smallest change to a single feature "
            "that would change the greenwashing classification."
        )

        # Compute counterfactuals for top features
        counterfactuals = []
        for feat in top_features[:15]:
            if feat not in feature_cols:
                continue

            feat_idx = feature_cols.index(feat)
            current_val = float(company_fm[feat]) if not pd.isna(company_fm[feat]) else 0.0
            feat_min = float(fm_full[feat].min())
            feat_max = float(fm_full[feat].max())
            feat_mean = float(fm_full[feat].mean())
            feat_std = float(fm_full[feat].std()) if fm_full[feat].std() > 0 else 1.0

            # Search for threshold that flips prediction
            best_cf = None
            target_class = 1 - current_pred

            # Try steps from current value toward population extremes
            if current_pred == 1:
                # Flagged -> try to make NOT flagged (reduce risk)
                test_values = np.linspace(current_val, feat_min, 30)
            else:
                # Not flagged -> try to make FLAGGED (increase risk)
                test_values = np.linspace(current_val, feat_max, 30)

            for test_val in test_values:
                X_test = X_current.copy()
                X_test[0, feat_idx] = test_val
                X_test_scaled = scaler.transform(X_test)
                test_pred = model.predict(X_test_scaled)[0]
                test_prob = model.predict_proba(X_test_scaled)[0][1]

                if test_pred == target_class:
                    change = test_val - current_val
                    change_pct = (change / (abs(current_val) + 1e-8)) * 100
                    counterfactuals.append({
                        'feature': feat,
                        'current_value': current_val,
                        'required_value': test_val,
                        'change': change,
                        'change_pct': change_pct,
                        'new_prediction': 'NOT FLAGGED' if target_class == 0 else 'FLAGGED',
                        'new_probability': test_prob,
                        'direction': 'decrease' if change < 0 else 'increase',
                        'change_in_std': change / feat_std,
                    })
                    break

        if counterfactuals:
            # Sort by smallest absolute change in std deviations
            counterfactuals.sort(key=lambda x: abs(x['change_in_std']))

            st.markdown(f"Found **{len(counterfactuals)} actionable counterfactuals** for {selected}:")

            # Easiest counterfactual highlight
            easiest = counterfactuals[0]
            direction_word = "decreased" if easiest['direction'] == 'decrease' else 'increased'
            arrow = "↓" if easiest['direction'] == 'decrease' else "↑"

            if current_pred == 1:
                st.markdown(
                    f'<div class="cf-result-good">'
                    f'<h3 style="color:#16a34a;margin:0;">EASIEST PATH TO LOW RISK</h3>'
                    f'<p style="font-size:18px;margin:10px 0;">If <b>{easiest["feature"]}</b> '
                    f'{direction_word} from <b>{easiest["current_value"]:.4f}</b> to '
                    f'<b>{easiest["required_value"]:.4f}</b> {arrow} → '
                    f'Risk becomes <b style="color:#16a34a;">{easiest["new_prediction"]}</b></p>'
                    f'<p style="color:#666;font-size:13px;">Change: {easiest["change"]:+.4f} '
                    f'({easiest["change_in_std"]:+.2f} std deviations)</p></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="cf-result-bad">'
                    f'<h3 style="color:#dc2626;margin:0;">TIPPING POINT TO HIGH RISK</h3>'
                    f'<p style="font-size:18px;margin:10px 0;">If <b>{easiest["feature"]}</b> '
                    f'{direction_word} from <b>{easiest["current_value"]:.4f}</b> to '
                    f'<b>{easiest["required_value"]:.4f}</b> {arrow} → '
                    f'Would become <b style="color:#dc2626;">{easiest["new_prediction"]}</b></p>'
                    f'<p style="color:#666;font-size:13px;">Change: {easiest["change"]:+.4f} '
                    f'({easiest["change_in_std"]:+.2f} std deviations)</p></div>',
                    unsafe_allow_html=True)

            # All counterfactuals
            for cf in counterfactuals:
                arrow = "↓" if cf['direction'] == 'decrease' else "↑"
                st.markdown(
                    f'<div class="counterfactual-card">'
                    f'<b>{cf["feature"]}</b><br>'
                    f'<span class="cf-arrow">{cf["current_value"]:.4f} → {cf["required_value"]:.4f} {arrow}</span>'
                    f' &nbsp; (change: {cf["change"]:+.4f} | {cf["change_in_std"]:+.2f} std)<br>'
                    f'<span style="color:#666;">Result: {cf["new_prediction"]} '
                    f'(probability: {cf["new_probability"]:.1%})</span></div>',
                    unsafe_allow_html=True)

            # Counterfactual bar chart
            cf_df = pd.DataFrame(counterfactuals)
            fig = px.bar(
                cf_df, x='feature', y='change_in_std',
                color='change_in_std',
                color_continuous_scale='RdBu_r',
                title='Required Change (in std deviations) to Flip Prediction',
                labels={'change_in_std': 'Change (std devs)', 'feature': 'Feature'},
            )
            fig.update_layout(xaxis_tickangle=-45, height=450)
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No single-feature counterfactual found. The prediction may require "
                    "multi-feature changes to flip.")

    # ================================================================
    # TAB 2: WHAT-IF SENSITIVITY SLIDERS
    # ================================================================
    with tab2:
        st.subheader("What-If Analysis")
        st.markdown(
            "**Adjust feature values with sliders and watch the prediction change in real time.**  \n"
            "This simulates how changes in ESG metrics would affect greenwashing risk."
        )

        # Select features to adjust
        slider_features = st.multiselect(
            "Select features to adjust (max 8)",
            top_features[:15],
            default=top_features[:5],
            max_selections=8,
            key='whatif_features',
        )

        if not slider_features:
            st.info("Select features above to start the What-If analysis.")
        else:
            # Create sliders
            modified_values = {}
            st.markdown("##### Adjust values:")

            for feat in slider_features:
                current_val = float(company_fm[feat]) if not pd.isna(company_fm[feat]) else 0.0
                feat_min = float(fm_full[feat].quantile(0.01))
                feat_max = float(fm_full[feat].quantile(0.99))
                feat_mean = float(fm_full[feat].mean())

                # Handle edge cases
                if feat_min >= feat_max:
                    feat_min = current_val - 1
                    feat_max = current_val + 1

                step = (feat_max - feat_min) / 100

                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    new_val = st.slider(
                        f"{feat}",
                        min_value=float(feat_min),
                        max_value=float(feat_max),
                        value=float(np.clip(current_val, feat_min, feat_max)),
                        step=float(step),
                        key=f"slider_{feat}",
                    )
                with col2:
                    st.caption(f"Current: {current_val:.4f}")
                with col3:
                    change = new_val - current_val
                    if abs(change) > 0.0001:
                        st.caption(f"Change: {change:+.4f}")
                    else:
                        st.caption("No change")

                modified_values[feat] = new_val

            # Compute new prediction
            X_modified = X_current.copy()
            for feat, val in modified_values.items():
                if feat in feature_cols:
                    feat_idx = feature_cols.index(feat)
                    X_modified[0, feat_idx] = val

            X_modified_scaled = scaler.transform(X_modified)
            new_prob = model.predict_proba(X_modified_scaled)[0][1]
            new_pred = model.predict(X_modified_scaled)[0]
            new_risk_est = _compute_risk_score_from_features(
                {feat: modified_values.get(feat, company_fm[feat]) for feat in fm_full.columns
                 if feat in company_fm.index},
                fm_full)

            st.markdown("---")
            st.subheader("Live Prediction Result")

            # Before/After comparison
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("##### BEFORE (Current)")
                st.metric("GW Probability", f"{current_prob:.1%}")
                st.metric("Prediction", "FLAGGED" if current_pred == 1 else "NOT FLAGGED")
            with col2:
                st.markdown("##### CHANGE")
                prob_change = new_prob - current_prob
                st.metric("Probability Shift", f"{prob_change:+.1%}",
                          delta=f"{prob_change:+.1%}", delta_color="inverse")
                flipped = new_pred != current_pred
                if flipped:
                    st.success("PREDICTION FLIPPED!")
                else:
                    st.info("No flip")
            with col3:
                st.markdown("##### AFTER (Modified)")
                st.metric("GW Probability", f"{new_prob:.1%}")
                st.metric("Prediction", "FLAGGED" if new_pred == 1 else "NOT FLAGGED")

            # Gauge comparison
            col1, col2 = st.columns(2)
            for col, (title, prob, color) in zip(
                [col1, col2],
                [("Current", current_prob, "#3498db"), ("Modified", new_prob, "#e74c3c")]
            ):
                with col:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={'text': title},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 30], 'color': '#e8f5e9'},
                                {'range': [30, 60], 'color': '#fff3e0'},
                                {'range': [60, 100], 'color': '#ffebee'},
                            ],
                            'threshold': {
                                'line': {'color': 'black', 'width': 3},
                                'thickness': 0.8,
                                'value': 50,
                            },
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)

            # Feature changes table
            changes_data = []
            for feat in slider_features:
                old_val = float(company_fm[feat]) if not pd.isna(company_fm[feat]) else 0.0
                new_val = modified_values[feat]
                changes_data.append({
                    'Feature': feat,
                    'Original': round(old_val, 4),
                    'Modified': round(new_val, 4),
                    'Change': round(new_val - old_val, 4),
                })
            st.dataframe(pd.DataFrame(changes_data), use_container_width=True)

    # ================================================================
    # TAB 3: FEATURE SENSITIVITY ANALYSIS
    # ================================================================
    with tab3:
        st.subheader("Feature Sensitivity Analysis")
        st.markdown(
            "**How sensitive is this company's prediction to each feature?**  \n"
            "Each feature is varied across its population range while others stay fixed. "
            "Steep curves = high sensitivity."
        )

        # Select features to analyze
        sensitivity_features = st.multiselect(
            "Select features for sensitivity sweep",
            top_features[:15],
            default=top_features[:4],
            max_selections=6,
            key='sens_features',
        )

        if sensitivity_features:
            # Compute sensitivity curves
            fig = make_subplots(
                rows=len(sensitivity_features), cols=1,
                subplot_titles=[f"Sensitivity: {f}" for f in sensitivity_features],
                vertical_spacing=0.08,
            )

            for row_idx, feat in enumerate(sensitivity_features):
                feat_idx = feature_cols.index(feat) if feat in feature_cols else None
                if feat_idx is None:
                    continue

                current_val = float(company_fm[feat]) if not pd.isna(company_fm[feat]) else 0.0
                feat_min = float(fm_full[feat].quantile(0.02))
                feat_max = float(fm_full[feat].quantile(0.98))

                if feat_min >= feat_max:
                    continue

                sweep_values = np.linspace(feat_min, feat_max, 50)
                probabilities = []

                for val in sweep_values:
                    X_sweep = X_current.copy()
                    X_sweep[0, feat_idx] = val
                    X_sweep_scaled = scaler.transform(X_sweep)
                    prob = model.predict_proba(X_sweep_scaled)[0][1]
                    probabilities.append(prob)

                # Add probability curve
                fig.add_trace(
                    go.Scatter(
                        x=sweep_values, y=probabilities,
                        mode='lines', name=feat,
                        line=dict(width=3),
                        hovertemplate=f'{feat}: %{{x:.4f}}<br>GW Prob: %{{y:.1%}}<extra></extra>',
                    ),
                    row=row_idx + 1, col=1,
                )

                # Mark current value
                fig.add_trace(
                    go.Scatter(
                        x=[current_val], y=[current_prob],
                        mode='markers', name=f'{feat} (current)',
                        marker=dict(size=14, color='red', symbol='star',
                                    line=dict(width=2, color='black')),
                        showlegend=False,
                        hovertemplate=f'CURRENT: {current_val:.4f}<br>Prob: {current_prob:.1%}<extra></extra>',
                    ),
                    row=row_idx + 1, col=1,
                )

                # Decision boundary line at 50%
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                              opacity=0.5, row=row_idx + 1, col=1)

                fig.update_yaxes(title_text="GW Probability", range=[0, 1],
                                 row=row_idx + 1, col=1)
                fig.update_xaxes(title_text=feat, row=row_idx + 1, col=1)

            fig.update_layout(
                height=350 * len(sensitivity_features),
                showlegend=False,
                title_text=f"Feature Sensitivity Curves -- {selected}",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **How to read this chart:**
            - **Red star** = company's current position
            - **Steep curve** = prediction is highly sensitive to this feature
            - **Flat curve** = feature has little impact on prediction
            - **Dashed line at 50%** = decision boundary (above = flagged, below = not flagged)
            """)

            # Sensitivity magnitude summary
            st.subheader("Sensitivity Magnitude Summary")
            sens_data = []
            for feat in sensitivity_features:
                feat_idx = feature_cols.index(feat) if feat in feature_cols else None
                if feat_idx is None:
                    continue

                feat_min = float(fm_full[feat].quantile(0.02))
                feat_max = float(fm_full[feat].quantile(0.98))

                # Probability at extremes
                X_low = X_current.copy()
                X_low[0, feat_idx] = feat_min
                X_high = X_current.copy()
                X_high[0, feat_idx] = feat_max

                prob_low = model.predict_proba(scaler.transform(X_low))[0][1]
                prob_high = model.predict_proba(scaler.transform(X_high))[0][1]

                sens_data.append({
                    'Feature': feat,
                    'Prob at Min': f"{prob_low:.1%}",
                    'Prob at Max': f"{prob_high:.1%}",
                    'Sensitivity Range': f"{abs(prob_high - prob_low):.1%}",
                    'Current Value': f"{float(company_fm[feat]):.4f}" if not pd.isna(company_fm[feat]) else "0",
                    'Can Flip?': "YES" if (prob_low < 0.5 < prob_high) or (prob_high < 0.5 < prob_low) else "No",
                })

            sens_df = pd.DataFrame(sens_data)
            st.dataframe(sens_df, use_container_width=True)

            # Sensitivity bar chart
            sens_df['Range'] = [abs(float(r.strip('%'))) for r in sens_df['Sensitivity Range']]
            fig = px.bar(
                sens_df.sort_values('Range', ascending=True),
                x='Range', y='Feature', orientation='h',
                title='Feature Sensitivity Magnitude (probability range when varied)',
                labels={'Range': 'Sensitivity (%)', 'Feature': ''},
                color='Range', color_continuous_scale='YlOrRd',
                text='Sensitivity Range',
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(height=max(300, len(sensitivity_features) * 60))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select features above to run sensitivity analysis.")


# ============================================================================
# PAGE 10: COMPANY COMPARISON MODE
# ============================================================================

def page_company_comparison(data):
    """Side-by-side comparison of multiple companies vs industry average."""

    st.markdown("""
    <style>
    .compare-header {
        background: linear-gradient(135deg, #0f3460 0%, #533483 50%, #e94560 100%);
        padding: 25px; border-radius: 12px; margin-bottom: 20px;
    }
    .compare-header h1 { color: white; margin: 0; font-family: monospace; }
    .compare-header p { color: #e0c3fc; margin: 5px 0 0 0; font-size: 14px; }
    .winner-badge {
        background: linear-gradient(135deg, #ffd700, #ffaa00);
        color: #333; padding: 4px 14px; border-radius: 20px;
        font-weight: bold; font-size: 13px; display: inline-block;
    }
    .loser-badge {
        background: #e74c3c; color: white; padding: 4px 14px;
        border-radius: 20px; font-weight: bold; font-size: 13px;
        display: inline-block;
    }
    </style>
    <div class="compare-header">
        <h1>COMPANY COMPARISON MODE</h1>
        <p>Side-by-side ESG comparison | Company A vs B vs Industry Average</p>
    </div>
    """, unsafe_allow_html=True)

    df = data['risk_scores']
    fm = data['feature_matrix']

    if df.empty or fm.empty:
        st.warning("Data not available. Run `python model_pipeline.py` first.")
        return

    company_names = sorted(df['company_name'].dropna().unique().tolist())

    # --- Company Selector ---
    st.subheader("Select Companies to Compare")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        company_a = st.selectbox("Company A", company_names, index=0, key='cmp_a')
    with col2:
        default_b = min(1, len(company_names) - 1)
        company_b = st.selectbox("Company B", company_names, index=default_b, key='cmp_b')
    with col3:
        add_more = st.checkbox("Add Company C", value=False)

    company_c = None
    if add_more:
        default_c = min(2, len(company_names) - 1)
        company_c = st.selectbox("Company C", company_names, index=default_c, key='cmp_c')

    selected = [company_a, company_b]
    if company_c:
        selected.append(company_c)

    # Remove duplicates
    selected = list(dict.fromkeys(selected))

    if len(selected) < 2:
        st.warning("Please select at least 2 different companies.")
        return

    # Gather data for each company
    companies_data = []
    for name in selected:
        risk_row = df[df['company_name'] == name]
        fm_row = fm[fm['company_name'] == name]
        if risk_row.empty or fm_row.empty:
            continue
        companies_data.append({
            'name': name,
            'risk': risk_row.iloc[0],
            'fm': fm_row.iloc[0],
        })

    if len(companies_data) < 2:
        st.error("Could not find data for selected companies.")
        return

    # Determine sector for industry average
    sectors = [c['risk']['sector'] for c in companies_data]
    primary_sector = max(set(sectors), key=sectors.count)
    sector_df = df[df['sector'] == primary_sector]
    sector_fm = fm[fm['sector'] == primary_sector] if 'sector' in fm.columns else pd.DataFrame()

    st.markdown("---")

    # ===================================================================
    # SECTION 1: HEAD-TO-HEAD KPI CARDS
    # ===================================================================
    st.subheader("1. Head-to-Head Overview")

    cols = st.columns(len(companies_data) + 1)  # +1 for industry avg

    for i, comp in enumerate(companies_data):
        with cols[i]:
            risk_score = float(comp['risk']['risk_score'])
            tier = comp['risk'].get('risk_tier', 'N/A')
            # Find best/worst
            is_lowest_risk = risk_score == min(c['risk']['risk_score'] for c in companies_data)
            is_highest_risk = risk_score == max(c['risk']['risk_score'] for c in companies_data)

            badge = ""
            if is_lowest_risk and len(companies_data) > 1:
                badge = '<span class="winner-badge">LOWEST RISK</span>'
            elif is_highest_risk and len(companies_data) > 1:
                badge = '<span class="loser-badge">HIGHEST RISK</span>'

            st.markdown(f"##### {comp['name']}")
            if badge:
                st.markdown(badge, unsafe_allow_html=True)
            st.metric("Risk Score", f"{risk_score:.1f}/100")
            st.metric("Risk Tier", str(tier))
            st.metric("ESG Risk", f"{comp['risk'].get('total_esg_risk_score', 0):.1f}")
            st.metric("Controversy", f"{comp['risk'].get('controversy_score', 0):.1f}")
            st.metric("Sector", str(comp['risk'].get('sector', 'N/A')))

    # Industry average column
    with cols[-1]:
        st.markdown(f"##### {primary_sector} Average")
        st.markdown('<span class="winner-badge">BENCHMARK</span>', unsafe_allow_html=True)
        st.metric("Risk Score", f"{sector_df['risk_score'].mean():.1f}/100")
        st.metric("Risk Tier", "Benchmark")
        st.metric("ESG Risk", f"{sector_df['total_esg_risk_score'].mean():.1f}" if 'total_esg_risk_score' in sector_df.columns else "N/A")
        st.metric("Controversy", f"{sector_df['controversy_score'].mean():.1f}" if 'controversy_score' in sector_df.columns else "N/A")
        st.metric("Companies", f"{len(sector_df)}")

    st.markdown("---")

    # ===================================================================
    # SECTION 2: RISK SCORE COMPARISON BAR CHART
    # ===================================================================
    st.subheader("2. Risk Score Comparison")

    bar_data = []
    for comp in companies_data:
        bar_data.append({
            'Company': comp['name'],
            'Risk Score': float(comp['risk']['risk_score']),
            'Type': 'Company',
        })
    bar_data.append({
        'Company': f'{primary_sector} Avg',
        'Risk Score': float(sector_df['risk_score'].mean()),
        'Type': 'Industry Average',
    })
    bar_data.append({
        'Company': 'Population Avg',
        'Risk Score': float(df['risk_score'].mean()),
        'Type': 'Population Average',
    })

    bar_df = pd.DataFrame(bar_data)
    colors = {'Company': '#e94560', 'Industry Average': '#3498db', 'Population Average': '#95a5a6'}

    fig = px.bar(
        bar_df, x='Company', y='Risk Score', color='Type',
        color_discrete_map=colors,
        title='Greenwashing Risk Score Comparison',
        text='Risk Score',
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(yaxis_range=[0, bar_df['Risk Score'].max() * 1.2], height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ===================================================================
    # SECTION 3: ESG PILLAR RADAR COMPARISON
    # ===================================================================
    st.subheader("3. ESG Pillar Profile Comparison")

    radar_features = [
        ('total_esg_risk_score', 'Total ESG Risk'),
        ('env_risk_score', 'Environmental'),
        ('social_risk_score', 'Social'),
        ('gov_risk_score', 'Governance'),
        ('controversy_score', 'Controversy'),
    ]
    available_radar = [(col, label) for col, label in radar_features if col in fm.columns]

    if available_radar:
        fig = go.Figure()
        radar_colors = ['#e94560', '#3498db', '#2ecc71', '#ff9800', '#9b59b6']
        categories = [label for _, label in available_radar]
        categories_closed = categories + [categories[0]]

        for i, comp in enumerate(companies_data):
            vals = []
            for col, _ in available_radar:
                cmin, cmax = fm[col].min(), fm[col].max()
                val = (float(comp['fm'].get(col, 0)) - cmin) / (cmax - cmin + 1e-8)
                vals.append(round(val, 3))
            vals_closed = vals + [vals[0]]

            fig.add_trace(go.Scatterpolar(
                r=vals_closed, theta=categories_closed,
                fill='toself', name=comp['name'],
                line=dict(color=radar_colors[i % len(radar_colors)], width=3),
                opacity=0.7,
            ))

        # Industry average
        if not sector_fm.empty:
            avg_vals = []
            for col, _ in available_radar:
                if col in sector_fm.columns:
                    cmin, cmax = fm[col].min(), fm[col].max()
                    val = (float(sector_fm[col].mean()) - cmin) / (cmax - cmin + 1e-8)
                    avg_vals.append(round(val, 3))
                else:
                    avg_vals.append(0)
            avg_vals_closed = avg_vals + [avg_vals[0]]

            fig.add_trace(go.Scatterpolar(
                r=avg_vals_closed, theta=categories_closed,
                fill='toself', name=f'{primary_sector} Avg',
                line=dict(color='#95a5a6', dash='dash', width=2),
                opacity=0.4,
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title='ESG Profile Radar (normalized 0-1)',
            height=550,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ===================================================================
    # SECTION 4: RISK COMPONENT BREAKDOWN COMPARISON
    # ===================================================================
    st.subheader("4. Risk Score Component Breakdown")

    comp_cols = {
        'comp_proxy': 'Proxy Score (40%)',
        'comp_linguistic': 'Linguistic Signal (15%)',
        'comp_divergence': 'ESG-Controversy Gap (15%)',
        'comp_credibility_inv': 'Low Credibility (15%)',
        'comp_controversy_ratio': 'Controversy Ratio (15%)',
    }

    comp_data = []
    for comp in companies_data:
        for col, label in comp_cols.items():
            val = comp['risk'].get(col, 0)
            comp_data.append({
                'Company': comp['name'],
                'Component': label,
                'Score': float(val) if pd.notna(val) else 0.0,
            })

    comp_df = pd.DataFrame(comp_data)

    fig = px.bar(
        comp_df, x='Component', y='Score', color='Company',
        barmode='group',
        title='Risk Score Components -- Which areas drive each company\'s risk?',
        color_discrete_sequence=['#e94560', '#3498db', '#2ecc71'],
        text='Score',
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(yaxis_range=[0, 110], height=500, xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ===================================================================
    # SECTION 5: NLP & LINGUISTIC COMPARISON
    # ===================================================================
    st.subheader("5. NLP & Linguistic Profile Comparison")

    nlp_metrics = [
        ('greenwashing_signal_score', 'GW Signal Score', 'Higher = more greenwashing language'),
        ('vague_language_count', 'Vague Language', 'Vague promises count'),
        ('concrete_evidence_count', 'Concrete Evidence', 'Measurable data points'),
        ('hedge_language_count', 'Hedge Language', 'Hedging/uncertain terms'),
        ('superlative_count', 'Superlatives', 'Unsubstantiated claims'),
        ('future_language_count', 'Future Language', 'Forward-looking promises'),
        ('text_polarity', 'Sentiment Polarity', 'Positive bias in text'),
        ('flesch_reading_ease', 'Readability', 'Higher = easier to read'),
        ('lexical_diversity', 'Lexical Diversity', 'Vocabulary richness'),
        ('aggregate_esg_nlp_score', 'Aggregate ESG NLP Score', '0-100 composite credibility'),
        ('regulatory_readiness_score', 'Regulatory Readiness', 'Policy framework alignment'),
        ('policy_esg_gap', 'Policy-ESG Gap', 'ESG talk vs policy grounding'),
        ('narrative_credibility_index', 'Narrative Credibility', 'Factual vs promotional'),
        ('commitment_credibility_score', 'Commitment Credibility', 'Past achievements vs promises'),
        ('multi_signal_greenwashing_score', 'Multi-Signal GW', 'Ensemble GW indicator'),
    ]

    nlp_data = []
    for comp in companies_data:
        for col, label, _ in nlp_metrics:
            val = comp['fm'].get(col, 0)
            nlp_data.append({
                'Company': comp['name'],
                'Metric': label,
                'Value': float(val) if pd.notna(val) else 0.0,
            })
    # Add industry average
    for col, label, _ in nlp_metrics:
        if col in sector_fm.columns and not sector_fm.empty:
            nlp_data.append({
                'Company': f'{primary_sector} Avg',
                'Metric': label,
                'Value': float(sector_fm[col].mean()),
            })

    nlp_df = pd.DataFrame(nlp_data)

    # Grouped bar chart
    fig = px.bar(
        nlp_df, x='Metric', y='Value', color='Company',
        barmode='group',
        title='NLP Linguistic Features -- Who uses more greenwashing language?',
        color_discrete_sequence=['#e94560', '#3498db', '#2ecc71', '#95a5a6'],
    )
    fig.update_layout(height=500, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed NLP table
    nlp_table = []
    for col, label, desc in nlp_metrics:
        row = {'Metric': label, 'Description': desc}
        for comp in companies_data:
            val = comp['fm'].get(col, 0)
            row[comp['name']] = f"{float(val):.4f}" if pd.notna(val) else "0"
        if col in sector_fm.columns and not sector_fm.empty:
            row[f'{primary_sector} Avg'] = f"{sector_fm[col].mean():.4f}"
        nlp_table.append(row)

    st.dataframe(pd.DataFrame(nlp_table), use_container_width=True)

    st.markdown("---")

    # ===================================================================
    # SECTION 6: KEY INDICATORS COMPARISON
    # ===================================================================
    st.subheader("6. Key Greenwashing Indicators")

    indicators = [
        ('controversy_risk_ratio', 'Controversy-Risk Ratio'),
        ('esg_controversy_divergence', 'ESG-Controversy Divergence'),
        ('risk_controversy_mismatch', 'Risk-Controversy Mismatch'),
        ('combined_anomaly_score', 'Combined Anomaly Score'),
        ('pillar_imbalance_score', 'Pillar Imbalance'),
        ('greenwashing_signal_score', 'GW Linguistic Score'),
    ]

    ind_data = []
    for comp in companies_data:
        for col, label in indicators:
            val = comp['fm'].get(col, 0)
            threshold = fm[col].quantile(0.75) if col in fm.columns else 0
            ind_data.append({
                'Company': comp['name'],
                'Indicator': label,
                'Value': float(val) if pd.notna(val) else 0.0,
                'Threshold (75th)': float(threshold),
                'Status': 'HIGH' if (pd.notna(val) and float(val) > float(threshold)) else 'Normal',
            })

    ind_df = pd.DataFrame(ind_data)

    # Heatmap-style indicator comparison
    pivot = ind_df.pivot(index='Indicator', columns='Company', values='Value')
    fig = px.imshow(
        pivot, text_auto='.3f',
        title='Indicator Heatmap (darker = higher value)',
        color_continuous_scale='RdYlGn_r',
        aspect='auto',
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Status table
    status_pivot = ind_df.pivot(index='Indicator', columns='Company', values='Status')
    st.markdown("**Threshold Status** (HIGH = above 75th percentile):")
    st.dataframe(status_pivot, use_container_width=True)

    # Count high indicators per company
    high_counts = ind_df[ind_df['Status'] == 'HIGH'].groupby('Company').size().reset_index(name='High Indicators')
    for _, row in high_counts.iterrows():
        if row['High Indicators'] >= 4:
            st.error(f"**{row['Company']}** has **{row['High Indicators']}/6** indicators above threshold -- strong greenwashing signal.")
        elif row['High Indicators'] >= 2:
            st.warning(f"**{row['Company']}** has **{row['High Indicators']}/6** indicators above threshold.")

    st.markdown("---")

    # ===================================================================
    # SECTION 7: SECTOR PEER POSITIONING
    # ===================================================================
    st.subheader("7. Sector Peer Positioning")

    # Scatter plot: ESG Risk vs Controversy with selected companies highlighted
    scatter_df = df.copy()
    if 'total_esg_risk_score' in scatter_df.columns and 'controversy_score' in scatter_df.columns:
        scatter_df['Highlight'] = scatter_df['company_name'].apply(
            lambda x: x if x in selected else 'Other Companies'
        )

        fig = px.scatter(
            scatter_df, x='total_esg_risk_score', y='controversy_score',
            color='Highlight',
            size='risk_score',
            hover_name='company_name',
            hover_data=['sector', 'risk_tier', 'risk_score'],
            title='All 480 Companies -- Selected Companies Highlighted',
            labels={'total_esg_risk_score': 'Total ESG Risk', 'controversy_score': 'Controversy'},
            size_max=20,
        )
        # Make "Other Companies" gray and small
        fig.update_layout(height=550)
        for trace in fig.data:
            if trace.name == 'Other Companies':
                trace.marker.color = '#ddd'
                trace.marker.size = 5
                trace.marker.opacity = 0.3
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ===================================================================
    # SECTION 8: SIDE-BY-SIDE SUMMARY TABLE
    # ===================================================================
    st.subheader("8. Complete Comparison Table")

    all_metrics = [
        ('risk_score', 'Greenwashing Risk Score', 'risk'),
        ('total_esg_risk_score', 'Total ESG Risk Score', 'risk'),
        ('controversy_score', 'Controversy Score', 'risk'),
        ('risk_tier', 'Risk Tier', 'risk'),
        ('gw_proxy_score', 'Proxy Score (0-5)', 'risk'),
        ('env_risk_score', 'Environmental Risk', 'fm'),
        ('social_risk_score', 'Social Risk', 'fm'),
        ('gov_risk_score', 'Governance Risk', 'fm'),
        ('pillar_imbalance_score', 'Pillar Imbalance', 'fm'),
        ('controversy_risk_ratio', 'Controversy-Risk Ratio', 'fm'),
        ('esg_controversy_divergence', 'ESG-Controversy Divergence', 'fm'),
        ('greenwashing_signal_score', 'GW Linguistic Score', 'fm'),
        ('vague_language_count', 'Vague Language Count', 'fm'),
        ('concrete_evidence_count', 'Concrete Evidence Count', 'fm'),
        ('text_polarity', 'Text Sentiment', 'fm'),
        ('flesch_reading_ease', 'Readability (Flesch)', 'fm'),
        ('combined_anomaly_score', 'Anomaly Score', 'fm'),
        ('aggregate_esg_nlp_score', 'Aggregate ESG NLP Score', 'fm'),
        ('regulatory_readiness_score', 'Regulatory Readiness', 'fm'),
        ('policy_esg_gap', 'Policy-ESG Gap', 'fm'),
        ('narrative_credibility_index', 'Narrative Credibility', 'fm'),
        ('commitment_credibility_score', 'Commitment Credibility', 'fm'),
        ('multi_signal_greenwashing_score', 'Multi-Signal GW Score', 'fm'),
        ('temporal_greenwashing_signal', 'Temporal GW Signal', 'fm'),
        ('news_greenwashing_signal', 'News GW Signal', 'fm'),
    ]

    table_rows = []
    for col, label, source in all_metrics:
        row = {'Metric': label}
        for comp in companies_data:
            data_source = comp['risk'] if source == 'risk' else comp['fm']
            val = data_source.get(col, None)
            if val is not None and not isinstance(val, str):
                try:
                    row[comp['name']] = f"{float(val):.4f}" if not pd.isna(val) else 'N/A'
                except (ValueError, TypeError):
                    row[comp['name']] = str(val)
            elif val is not None:
                row[comp['name']] = str(val)
            else:
                row[comp['name']] = 'N/A'

        # Industry average
        if source == 'risk' and col in sector_df.columns:
            try:
                row[f'{primary_sector} Avg'] = f"{sector_df[col].mean():.4f}"
            except TypeError:
                row[f'{primary_sector} Avg'] = 'N/A'
        elif source == 'fm' and col in sector_fm.columns:
            try:
                row[f'{primary_sector} Avg'] = f"{sector_fm[col].mean():.4f}"
            except TypeError:
                row[f'{primary_sector} Avg'] = 'N/A'
        else:
            row[f'{primary_sector} Avg'] = 'N/A'

        table_rows.append(row)

    summary_df = pd.DataFrame(table_rows)
    st.dataframe(summary_df, use_container_width=True, height=600)

    # --- Export comparison ---
    csv_export = summary_df.to_csv(index=False)
    st.download_button(
        label="Download Comparison (CSV)",
        data=csv_export,
        file_name=f"ESG_Comparison_{'_vs_'.join(selected)}.csv",
        mime="text/csv",
    )


# ============================================================================
# PAGE 11: TIME-SERIES RISK TRACKING
# ============================================================================

@st.cache_data
def _load_timeseries_data():
    """Load the ESG financial time-series dataset (11 years, 1000 companies)."""
    ts_path = os.path.join(BASE_DIR, "data", "company_esg_financial_dataset.csv")
    if os.path.exists(ts_path):
        df = pd.read_csv(ts_path)
        return df
    return pd.DataFrame()


def _compute_simulated_risk_history(company_fm, fm_full, ts_company_df):
    """
    Compute a simulated risk score history by combining:
    1. Actual ESG time-series data (ESG_Overall, E/S/G, Carbon, etc.)
    2. Risk score model calibrated to the current known risk score.
    """
    if ts_company_df.empty:
        return pd.DataFrame()

    ts = ts_company_df.sort_values('Year').copy()

    # Normalize ESG scores to 0-100 risk (invert: higher ESG = lower risk)
    esg_max = 100
    ts['esg_risk_proxy'] = esg_max - ts['ESG_Overall']

    # Carbon intensity as controversy proxy
    ts['carbon_intensity'] = ts['CarbonEmissions'] / (ts['Revenue'] + 1e-8)
    ci_min, ci_max = ts['carbon_intensity'].min(), ts['carbon_intensity'].max()
    ts['controversy_proxy'] = ((ts['carbon_intensity'] - ci_min) / (ci_max - ci_min + 1e-8)) * 5

    # Pillar imbalance
    ts['pillar_imbalance'] = ts[['ESG_Environmental', 'ESG_Social', 'ESG_Governance']].std(axis=1)

    # Composite risk score (calibrated)
    ts['divergence'] = ts['controversy_proxy'] - (ts['esg_risk_proxy'] / 20)
    ts['raw_risk'] = (
        0.35 * ts['esg_risk_proxy'] +
        0.25 * (ts['controversy_proxy'] * 20) +
        0.20 * (ts['pillar_imbalance'] * 3) +
        0.20 * (ts['divergence'].clip(0) * 15)
    )

    # Scale to 0-100
    r_min, r_max = ts['raw_risk'].min(), ts['raw_risk'].max()
    if r_max > r_min:
        ts['risk_score'] = ((ts['raw_risk'] - r_min) / (r_max - r_min)) * 80 + 10
    else:
        ts['risk_score'] = 50.0

    return ts


def _forecast_risk(ts_df, forecast_years=3):
    """Simple linear regression forecast for risk score trend."""
    if len(ts_df) < 3:
        return pd.DataFrame()

    from sklearn.linear_model import LinearRegression

    X = ts_df['Year'].values.reshape(-1, 1)
    y = ts_df['risk_score'].values

    model = LinearRegression()
    model.fit(X, y)

    last_year = int(ts_df['Year'].max())
    future_years = np.arange(last_year + 1, last_year + 1 + forecast_years).reshape(-1, 1)
    future_risk = model.predict(future_years)

    # Confidence interval (simple std-based)
    residuals = y - model.predict(X)
    std_err = np.std(residuals)

    forecast = pd.DataFrame({
        'Year': future_years.flatten(),
        'risk_score': np.clip(future_risk, 0, 100),
        'lower_bound': np.clip(future_risk - 1.96 * std_err, 0, 100),
        'upper_bound': np.clip(future_risk + 1.96 * std_err, 0, 100),
        'type': 'Forecast',
    })

    return forecast


def page_timeseries_tracking(data):
    """Time-Series Risk Tracking: historical trends, before/after, forecasting."""

    st.markdown("""
    <style>
    .ts-header {
        background: linear-gradient(135deg, #1b2838 0%, #2a4858 50%, #1a6b4a 100%);
        padding: 25px; border-radius: 12px; margin-bottom: 20px;
    }
    .ts-header h1 { color: #4ade80; margin: 0; font-family: monospace; }
    .ts-header p { color: #a7f3d0; margin: 5px 0 0 0; font-size: 14px; }
    .trend-up { color: #ef4444; font-weight: bold; }
    .trend-down { color: #22c55e; font-weight: bold; }
    .trend-stable { color: #6b7280; font-weight: bold; }
    </style>
    <div class="ts-header">
        <h1>TIME-SERIES RISK TRACKING</h1>
        <p>Historical ESG trends (2015-2025) | Before vs After | Trend Prediction</p>
    </div>
    """, unsafe_allow_html=True)

    risk_df = data['risk_scores']
    fm = data['feature_matrix']

    if risk_df.empty:
        st.warning("Risk scores not available. Run `python model_pipeline.py` first.")
        return

    # Load time-series data
    ts_data = _load_timeseries_data()

    if ts_data.empty:
        st.error("Time-series data not found at `data/company_esg_financial_dataset.csv`.")
        return

    ts_companies = sorted(ts_data['CompanyName'].unique().tolist())

    # === COMPANY SELECTOR ===
    st.subheader("Select Company")
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox("Choose a company from the time-series dataset",
                                ts_companies, key='ts_company')
    with col2:
        forecast_years = st.slider("Forecast years", 1, 5, 3, key='ts_forecast')

    if not selected:
        return

    ts_company = ts_data[ts_data['CompanyName'] == selected].copy()
    ts_history = _compute_simulated_risk_history(None, fm, ts_company)

    if ts_history.empty:
        st.error(f"No time-series data available for {selected}.")
        return

    # Forecast
    forecast = _forecast_risk(ts_history, forecast_years)

    st.markdown("---")

    # ===================================================================
    # SECTION 1: RISK SCORE OVER TIME
    # ===================================================================
    st.subheader("1. Risk Score Over Time")

    # Combine history + forecast
    history_plot = ts_history[['Year', 'risk_score']].copy()
    history_plot['type'] = 'Historical'
    history_plot['lower_bound'] = history_plot['risk_score']
    history_plot['upper_bound'] = history_plot['risk_score']

    if not forecast.empty:
        # Add last historical point to forecast for seamless line
        bridge = history_plot.iloc[[-1]].copy()
        bridge['type'] = 'Forecast'
        combined = pd.concat([history_plot, bridge, forecast], ignore_index=True)
    else:
        combined = history_plot.copy()

    fig = go.Figure()

    # Historical line
    hist = combined[combined['type'] == 'Historical']
    fig.add_trace(go.Scatter(
        x=hist['Year'], y=hist['risk_score'],
        mode='lines+markers', name='Historical Risk Score',
        line=dict(color='#e94560', width=3),
        marker=dict(size=8),
        hovertemplate='Year: %{x}<br>Risk Score: %{y:.1f}<extra></extra>',
    ))

    # Forecast line + confidence band
    if not forecast.empty:
        fc = combined[combined['type'] == 'Forecast']
        fig.add_trace(go.Scatter(
            x=fc['Year'], y=fc['risk_score'],
            mode='lines+markers', name='Forecast',
            line=dict(color='#3b82f6', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond'),
            hovertemplate='Year: %{x}<br>Forecast: %{y:.1f}<extra></extra>',
        ))
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=list(fc['Year']) + list(fc['Year'][::-1]),
            y=list(fc['upper_bound']) + list(fc['lower_bound'][::-1]),
            fill='toself', fillcolor='rgba(59,130,246,0.15)',
            line=dict(color='rgba(0,0,0,0)'),
            name='95% Confidence',
            hoverinfo='skip',
        ))

    # Risk tier backgrounds
    fig.add_hrect(y0=0, y1=20, fillcolor='#22c55e', opacity=0.07, annotation_text='Very Low',
                  annotation_position='top left')
    fig.add_hrect(y0=20, y1=40, fillcolor='#84cc16', opacity=0.05)
    fig.add_hrect(y0=40, y1=60, fillcolor='#eab308', opacity=0.05)
    fig.add_hrect(y0=60, y1=80, fillcolor='#f97316', opacity=0.05)
    fig.add_hrect(y0=80, y1=100, fillcolor='#ef4444', opacity=0.07, annotation_text='Very High',
                  annotation_position='top left')

    fig.update_layout(
        title=f'Greenwashing Risk Score Timeline -- {selected}',
        xaxis_title='Year', yaxis_title='Risk Score (0-100)',
        yaxis_range=[0, 100], height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Trend stats
    first_risk = float(ts_history.iloc[0]['risk_score'])
    last_risk = float(ts_history.iloc[-1]['risk_score'])
    change = last_risk - first_risk
    first_year = int(ts_history.iloc[0]['Year'])
    last_year = int(ts_history.iloc[-1]['Year'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"Risk ({first_year})", f"{first_risk:.1f}")
    with col2:
        st.metric(f"Risk ({last_year})", f"{last_risk:.1f}",
                  delta=f"{change:+.1f}", delta_color="inverse")
    with col3:
        trend = "INCREASING" if change > 2 else "DECREASING" if change < -2 else "STABLE"
        trend_class = "trend-up" if change > 2 else "trend-down" if change < -2 else "trend-stable"
        st.markdown(f"**Trend:** <span class='{trend_class}'>{trend}</span>", unsafe_allow_html=True)
    with col4:
        if not forecast.empty:
            future_risk = float(forecast.iloc[-1]['risk_score'])
            future_year = int(forecast.iloc[-1]['Year'])
            st.metric(f"Forecast ({future_year})", f"{future_risk:.1f}",
                      delta=f"{future_risk - last_risk:+.1f} from now", delta_color="inverse")

    st.markdown("---")

    # ===================================================================
    # SECTION 2: ESG PILLAR TRENDS
    # ===================================================================
    st.subheader("2. ESG Pillar Trends (2015-2025)")

    fig = go.Figure()
    pillars = [
        ('ESG_Environmental', 'Environmental', '#22c55e'),
        ('ESG_Social', 'Social', '#3b82f6'),
        ('ESG_Governance', 'Governance', '#a855f7'),
        ('ESG_Overall', 'Overall ESG', '#ef4444'),
    ]
    for col, label, color in pillars:
        if col in ts_history.columns:
            fig.add_trace(go.Scatter(
                x=ts_history['Year'], y=ts_history[col],
                mode='lines+markers', name=label,
                line=dict(color=color, width=2),
                hovertemplate=f'{label}: %{{y:.1f}}<extra></extra>',
            ))

    fig.update_layout(
        title=f'ESG Pillar Scores Over Time -- {selected}',
        xaxis_title='Year', yaxis_title='ESG Score',
        height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Pillar gap trend
    if 'pillar_imbalance' in ts_history.columns:
        fig = px.area(
            ts_history, x='Year', y='pillar_imbalance',
            title=f'Pillar Imbalance Over Time -- {selected} (higher = more uneven ESG profile)',
            labels={'pillar_imbalance': 'Imbalance (std across E/S/G)'},
            color_discrete_sequence=['#f97316'],
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ===================================================================
    # SECTION 3: BEFORE VS AFTER CONTROVERSY
    # ===================================================================
    st.subheader("3. Before vs After Controversy Analysis")
    st.markdown("Select a **pivot year** to compare ESG metrics before and after a controversy event:")

    years = sorted(ts_history['Year'].unique().tolist())
    if len(years) >= 3:
        pivot_year = st.select_slider(
            "Pivot Year (controversy event)",
            options=years[1:-1],
            value=years[len(years) // 2],
            key='ts_pivot',
        )

        before = ts_history[ts_history['Year'] < pivot_year]
        after = ts_history[ts_history['Year'] >= pivot_year]

        if not before.empty and not after.empty:
            # Before vs After metrics
            metrics_compare = [
                ('risk_score', 'Risk Score'),
                ('ESG_Overall', 'ESG Overall'),
                ('ESG_Environmental', 'Environmental'),
                ('ESG_Social', 'Social'),
                ('ESG_Governance', 'Governance'),
                ('controversy_proxy', 'Controversy Proxy'),
                ('CarbonEmissions', 'Carbon Emissions'),
                ('pillar_imbalance', 'Pillar Imbalance'),
            ]

            ba_data = []
            for col, label in metrics_compare:
                if col in before.columns:
                    before_avg = float(before[col].mean())
                    after_avg = float(after[col].mean())
                    change_pct = ((after_avg - before_avg) / (abs(before_avg) + 1e-8)) * 100
                    ba_data.append({
                        'Metric': label,
                        'Before (avg)': round(before_avg, 2),
                        'After (avg)': round(after_avg, 2),
                        'Change': round(after_avg - before_avg, 2),
                        'Change %': round(change_pct, 1),
                    })

            ba_df = pd.DataFrame(ba_data)

            # Grouped bar chart
            ba_melt = ba_df.melt(id_vars='Metric', value_vars=['Before (avg)', 'After (avg)'],
                                 var_name='Period', value_name='Value')
            fig = px.bar(
                ba_melt, x='Metric', y='Value', color='Period',
                barmode='group',
                color_discrete_map={'Before (avg)': '#3b82f6', 'After (avg)': '#ef4444'},
                title=f'Before vs After {pivot_year} -- Key Metrics',
            )
            fig.update_layout(height=450, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

            # Change waterfall
            fig = go.Figure(go.Waterfall(
                x=ba_df['Metric'],
                y=ba_df['Change %'],
                textposition='outside',
                text=[f"{v:+.1f}%" for v in ba_df['Change %']],
                connector=dict(line=dict(color='#666')),
                increasing=dict(marker=dict(color='#ef4444')),
                decreasing=dict(marker=dict(color='#22c55e')),
            ))
            fig.update_layout(
                title=f'Percentage Change: After {pivot_year} vs Before',
                yaxis_title='Change (%)', height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.dataframe(ba_df, use_container_width=True)

            # Narrative
            risk_change = ba_df[ba_df['Metric'] == 'Risk Score']['Change'].values[0]
            if risk_change > 5:
                st.error(f"Risk score **increased by {risk_change:.1f} points** after {pivot_year}. "
                         f"This suggests a potential controversy event or ESG deterioration.")
            elif risk_change < -5:
                st.success(f"Risk score **decreased by {abs(risk_change):.1f} points** after {pivot_year}. "
                           f"ESG performance appears to have improved.")
            else:
                st.info(f"Risk score remained relatively stable around {pivot_year} "
                        f"(change: {risk_change:+.1f}).")

    st.markdown("---")

    # ===================================================================
    # SECTION 4: CARBON & ENVIRONMENTAL TRENDS
    # ===================================================================
    st.subheader("4. Environmental Impact Trends")

    col1, col2 = st.columns(2)
    with col1:
        if 'CarbonEmissions' in ts_history.columns:
            fig = px.bar(
                ts_history, x='Year', y='CarbonEmissions',
                title=f'Carbon Emissions Over Time -- {selected}',
                color='CarbonEmissions', color_continuous_scale='Reds',
                text='CarbonEmissions',
            )
            fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'carbon_intensity' in ts_history.columns:
            fig = px.line(
                ts_history, x='Year', y='carbon_intensity',
                title=f'Carbon Intensity (Emissions/Revenue) -- {selected}',
                markers=True,
                color_discrete_sequence=['#ef4444'],
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if 'EnergyConsumption' in ts_history.columns:
            fig = px.area(
                ts_history, x='Year', y='EnergyConsumption',
                title=f'Energy Consumption -- {selected}',
                color_discrete_sequence=['#f97316'],
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if 'WaterUsage' in ts_history.columns:
            fig = px.area(
                ts_history, x='Year', y='WaterUsage',
                title=f'Water Usage -- {selected}',
                color_discrete_sequence=['#3b82f6'],
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ===================================================================
    # SECTION 5: TREND PREDICTION & FORECASTING
    # ===================================================================
    st.subheader("5. Trend Prediction & Forecasting")

    if not forecast.empty:
        # Forecast summary
        last_hist = float(ts_history.iloc[-1]['risk_score'])
        last_forecast = float(forecast.iloc[-1]['risk_score'])
        forecast_change = last_forecast - last_hist

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Risk", f"{last_hist:.1f}")
        with col2:
            st.metric(f"Predicted ({int(forecast.iloc[-1]['Year'])})", f"{last_forecast:.1f}",
                      delta=f"{forecast_change:+.1f}", delta_color="inverse")
        with col3:
            direction = "WORSENING" if forecast_change > 2 else "IMPROVING" if forecast_change < -2 else "STABLE"
            dir_color = "#ef4444" if forecast_change > 2 else "#22c55e" if forecast_change < -2 else "#6b7280"
            st.markdown(f"**Forecast Direction:**")
            st.markdown(f"<span style='color:{dir_color};font-size:24px;font-weight:bold;'>{direction}</span>",
                        unsafe_allow_html=True)

        # Year-by-year forecast table
        fc_display = forecast[['Year', 'risk_score', 'lower_bound', 'upper_bound']].copy()
        fc_display.columns = ['Year', 'Predicted Risk', 'Lower Bound (95%)', 'Upper Bound (95%)']
        fc_display = fc_display.round(1)
        st.dataframe(fc_display, use_container_width=True)

        # Multi-metric forecast
        st.markdown("##### Multi-Metric Forecast")
        forecast_metrics = ['ESG_Overall', 'ESG_Environmental', 'CarbonEmissions']
        forecast_available = [m for m in forecast_metrics if m in ts_history.columns]

        if forecast_available:
            from sklearn.linear_model import LinearRegression

            fig = make_subplots(
                rows=len(forecast_available), cols=1,
                subplot_titles=[f'{m} Forecast' for m in forecast_available],
                vertical_spacing=0.1,
            )

            colors = ['#22c55e', '#3b82f6', '#ef4444']
            for idx, metric in enumerate(forecast_available):
                X = ts_history['Year'].values.reshape(-1, 1)
                y = ts_history[metric].values

                lr = LinearRegression()
                lr.fit(X, y)

                last_yr = int(ts_history['Year'].max())
                future_x = np.arange(last_yr + 1, last_yr + 1 + forecast_years).reshape(-1, 1)
                future_y = lr.predict(future_x)

                # Historical
                fig.add_trace(go.Scatter(
                    x=ts_history['Year'], y=ts_history[metric],
                    mode='lines+markers', name=f'{metric} (Historical)',
                    line=dict(color=colors[idx % 3], width=2),
                    showlegend=(idx == 0),
                ), row=idx + 1, col=1)

                # Forecast
                bridge_x = [last_yr] + list(future_x.flatten())
                bridge_y = [float(ts_history[metric].iloc[-1])] + list(future_y)
                fig.add_trace(go.Scatter(
                    x=bridge_x, y=bridge_y,
                    mode='lines+markers', name=f'{metric} (Forecast)',
                    line=dict(color=colors[idx % 3], width=2, dash='dash'),
                    marker=dict(symbol='diamond'),
                    showlegend=(idx == 0),
                ), row=idx + 1, col=1)

            fig.update_layout(height=350 * len(forecast_available), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ===================================================================
    # SECTION 6: INDUSTRY TREND COMPARISON
    # ===================================================================
    st.subheader("6. Industry Trend Comparison")

    industry = ts_company.iloc[0].get('Industry', None)
    if industry:
        industry_data = ts_data[ts_data['Industry'] == industry]
        industry_avg = industry_data.groupby('Year').agg({
            'ESG_Overall': 'mean',
            'CarbonEmissions': 'mean',
            'ESG_Environmental': 'mean',
        }).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts_history['Year'], y=ts_history['ESG_Overall'],
            mode='lines+markers', name=selected,
            line=dict(color='#e94560', width=3),
        ))
        fig.add_trace(go.Scatter(
            x=industry_avg['Year'], y=industry_avg['ESG_Overall'],
            mode='lines+markers', name=f'{industry} Average',
            line=dict(color='#3b82f6', width=2, dash='dash'),
        ))
        fig.update_layout(
            title=f'ESG Overall: {selected} vs {industry} Industry Average',
            xaxis_title='Year', yaxis_title='ESG Score',
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Gap analysis
        if not industry_avg.empty:
            merged = ts_history[['Year', 'ESG_Overall']].merge(
                industry_avg[['Year', 'ESG_Overall']],
                on='Year', suffixes=('_company', '_industry')
            )
            merged['gap'] = merged['ESG_Overall_company'] - merged['ESG_Overall_industry']

            fig = go.Figure()
            colors = ['#22c55e' if g >= 0 else '#ef4444' for g in merged['gap']]
            fig.add_trace(go.Bar(
                x=merged['Year'], y=merged['gap'],
                marker_color=colors,
                text=[f"{g:+.1f}" for g in merged['gap']],
                textposition='outside',
                hovertemplate='Year: %{x}<br>Gap: %{y:+.1f}<extra></extra>',
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title=f'ESG Gap: {selected} vs {industry} Average (green = above avg)',
                xaxis_title='Year', yaxis_title='ESG Score Gap',
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ===================================================================
    # SECTION 7: DATA TABLE + EXPORT
    # ===================================================================
    st.subheader("7. Complete Time-Series Data")

    display_cols = ['Year', 'ESG_Overall', 'ESG_Environmental', 'ESG_Social',
                    'ESG_Governance', 'CarbonEmissions', 'WaterUsage',
                    'EnergyConsumption', 'Revenue', 'ProfitMargin']
    available_display = [c for c in display_cols if c in ts_history.columns]

    if 'risk_score' in ts_history.columns:
        available_display = ['Year', 'risk_score'] + [c for c in available_display if c != 'Year']

    st.dataframe(ts_history[available_display].round(2), use_container_width=True)

    csv_export = ts_history[available_display].to_csv(index=False)
    st.download_button(
        label="Download Time-Series Data (CSV)",
        data=csv_export,
        file_name=f"ESG_TimeSeries_{selected}.csv",
        mime="text/csv",
    )


# ============================================================================
# PAGE 12: SHAP EXPLANATIONS (all interactive)
# ============================================================================

def page_shap_explanations(data):
    """SHAP explanations -- REAL SHAP values from regenerated model on full 205-feature set.

    Displays:
        1. Global SHAP feature importance (mean |SHAP value|)
        2. New ESG features (Categories 7-10) ranking and impact
        3. Per-company SHAP waterfall (force plot equivalent)
        4. Feature distribution explorer (linked to SHAP-ranked features)
        5. SHAP-based correlation heatmap
    """

    st.title("SHAP Explanations -- Why Companies Are Flagged")
    st.markdown(
        "**SHAP (SHapley Additive exPlanations)** shows the marginal contribution of each "
        "feature to a company's greenwashing prediction. Unlike basic feature importance, "
        "SHAP values are computed per-company and explain individual predictions."
    )

    # Load REAL SHAP data
    shap_fi = data.get('shap_feature_importance', pd.DataFrame())
    shap_all = data.get('shap_values_all', pd.DataFrame())
    fm = data['feature_matrix']

    # Fallback warning if SHAP not regenerated
    if shap_fi.empty or shap_all.empty:
        st.error(
            "**SHAP data not found.** Please run `python regenerate_shap.py` from the project root "
            "to generate SHAP values on the current feature matrix (with all 91 NLP features)."
        )
        st.info("This page expects two files:\n"
                "- `data/processed/shap_feature_importance.csv`\n"
                "- `data/processed/shap_values_all.csv`")
        return

    # Define which features are NEW (Categories 7-10)
    new_feature_set = {
        'paris_agreement_alignment', 'eu_taxonomy_alignment', 'sec_climate_alignment',
        'tcfd_alignment', 'sdg_alignment', 'gri_standards_alignment',
        'regulatory_breadth_index', 'total_policy_density', 'policy_specificity_score',
        'policy_esg_gap', 'framework_consistency_score', 'regulatory_readiness_score',
        'promotional_intent_density', 'defensive_intent_density', 'factual_intent_density',
        'strategic_intent_density', 'narrative_credibility_index', 'promotional_dominance_score',
        'intent_diversity_score', 'defensive_to_factual_ratio', 'sentiment_intent_divergence',
        'news_greenwashing_signal', 'past_achievement_density', 'present_action_density',
        'specific_future_density', 'vague_future_density', 'temporal_balance_score',
        'commitment_credibility_score', 'temporal_specificity_ratio', 'progress_to_promise_ratio',
        'year_mention_density', 'temporal_greenwashing_signal',
        'policy_sentiment_interaction', 'readability_greenwashing_interaction',
        'temporal_policy_interaction', 'vocabulary_intent_interaction',
        'evidence_readability_interaction', 'claim_credibility_ratio',
        'multi_signal_greenwashing_score', 'esg_linguistic_credibility_index',
        'credibility_confidence_interval', 'policy_temporal_alignment',
        'narrative_consistency_score', 'aggregate_esg_nlp_score',
    }

    # Status badge
    n_features = len(shap_fi)
    n_new_in_top50 = sum(1 for f in shap_fi.head(50)['feature'].tolist() if f in new_feature_set)
    n_new_total = sum(1 for f in shap_fi['feature'].tolist() if f in new_feature_set)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Features Analyzed", f"{n_features}")
    col_b.metric("New ESG Features", f"{n_new_total}", delta=f"+{n_new_total} vs original")
    col_c.metric("New Features in Top 50", f"{n_new_in_top50}/50")
    col_d.metric("Companies Explained", f"{len(shap_all)}")

    st.markdown("---")

    # ====================================================================
    # SECTION 1: GLOBAL SHAP FEATURE IMPORTANCE
    # ====================================================================
    st.subheader("1. Global SHAP Feature Importance (Top 25)")
    st.markdown(
        "Mean absolute SHAP value across all 480 companies. Higher = more impact on predictions. "
        "**Green = NEW feature from Categories 7-10 (Government Policy, News Intent, Temporal, Aggregate).**"
    )

    top25 = shap_fi.head(25).copy()
    top25['is_new'] = top25['feature'].isin(new_feature_set)
    top25['color'] = top25['is_new'].map({True: '#27ae60', False: '#e74c3c'})
    top25 = top25.sort_values('shap_importance', ascending=True)

    fig = go.Figure(go.Bar(
        x=top25['shap_importance'],
        y=top25['feature'],
        orientation='h',
        marker=dict(color=top25['color'].tolist()),
        text=top25['shap_importance'].round(4),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>SHAP Importance: %{x:.6f}<br>%{customdata}<extra></extra>',
        customdata=['NEW Cat 7-10' if x else 'Original' for x in top25['is_new']],
    ))
    fig.update_layout(
        title='Mean |SHAP value| -- impact on greenwashing prediction',
        xaxis_title='Mean |SHAP|',
        yaxis_title='Feature',
        height=700,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ====================================================================
    # SECTION 2: NEW FEATURES RANKING
    # ====================================================================
    st.markdown("---")
    st.subheader("2. New ESG Features (Categories 7-10) -- SHAP Impact")
    st.markdown(
        "How the 44 new features (Government Policy, News Intent, Temporal, Aggregate) "
        "contribute to greenwashing detection."
    )

    new_features_df = shap_fi[shap_fi['feature'].isin(new_feature_set)].copy()
    new_features_df = new_features_df.reset_index(drop=False).rename(columns={'index': 'global_rank'})
    new_features_df['global_rank'] = new_features_df['global_rank'] + 1

    if not new_features_df.empty:
        # Categorize each new feature
        def categorize(feat):
            policy = {'paris_agreement_alignment', 'eu_taxonomy_alignment', 'sec_climate_alignment',
                      'tcfd_alignment', 'sdg_alignment', 'gri_standards_alignment',
                      'regulatory_breadth_index', 'total_policy_density', 'policy_specificity_score',
                      'policy_esg_gap', 'framework_consistency_score', 'regulatory_readiness_score'}
            news = {'promotional_intent_density', 'defensive_intent_density', 'factual_intent_density',
                    'strategic_intent_density', 'narrative_credibility_index', 'promotional_dominance_score',
                    'intent_diversity_score', 'defensive_to_factual_ratio', 'sentiment_intent_divergence',
                    'news_greenwashing_signal'}
            temporal = {'past_achievement_density', 'present_action_density', 'specific_future_density',
                        'vague_future_density', 'temporal_balance_score', 'commitment_credibility_score',
                        'temporal_specificity_ratio', 'progress_to_promise_ratio', 'year_mention_density',
                        'temporal_greenwashing_signal'}
            if feat in policy:
                return 'Cat 7: Government Policy'
            elif feat in news:
                return 'Cat 8: News Intent'
            elif feat in temporal:
                return 'Cat 9: Temporal'
            else:
                return 'Cat 10: Aggregate'

        new_features_df['category'] = new_features_df['feature'].apply(categorize)
        new_features_df = new_features_df.sort_values('shap_importance', ascending=False)

        fig = px.bar(
            new_features_df.head(20),
            x='shap_importance',
            y='feature',
            color='category',
            orientation='h',
            title='Top 20 New ESG Features by SHAP Impact',
            labels={'shap_importance': 'Mean |SHAP|', 'feature': 'Feature'},
            color_discrete_map={
                'Cat 7: Government Policy': '#3498db',
                'Cat 8: News Intent': '#9b59b6',
                'Cat 9: Temporal': '#f39c12',
                'Cat 10: Aggregate': '#27ae60',
            },
            text='global_rank',
        )
        fig.update_traces(texttemplate='Rank #%{text}', textposition='outside')
        fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        # Category summary
        cat_summary = new_features_df.groupby('category').agg(
            num_features=('feature', 'count'),
            mean_shap=('shap_importance', 'mean'),
            max_shap=('shap_importance', 'max'),
            best_rank=('global_rank', 'min'),
        ).reset_index().sort_values('mean_shap', ascending=False)
        st.markdown("**Category Summary:**")
        st.dataframe(cat_summary, use_container_width=True, hide_index=True)
    else:
        st.warning("No new features found in SHAP importance file. Run `python regenerate_shap.py`.")

    # ====================================================================
    # SECTION 3: PER-COMPANY SHAP WATERFALL
    # ====================================================================
    st.markdown("---")
    st.subheader("3. Per-Company SHAP Waterfall (Force Plot)")
    st.markdown(
        "Shows which features pushed a SPECIFIC company's prediction up (toward greenwashing) or "
        "down (toward genuine ESG). This is the core of SHAP -- it explains individual predictions."
    )

    if not shap_all.empty and 'company_name' in shap_all.columns:
        company_list = sorted(shap_all['company_name'].dropna().unique().tolist())

        # Default to a high-risk company if available
        default_idx = 0
        if not data.get('risk_scores', pd.DataFrame()).empty:
            risk_df = data['risk_scores']
            high_risk = risk_df.nlargest(1, 'risk_score')['company_name'].tolist()
            if high_risk and high_risk[0] in company_list:
                default_idx = company_list.index(high_risk[0])

        selected = st.selectbox(
            "Select a company to explain",
            company_list,
            index=default_idx,
            key='shap_company_selector',
        )

        if selected:
            company_shap = shap_all[shap_all['company_name'] == selected].iloc[0]
            shap_features = [c for c in shap_all.columns if c != 'company_name']
            shap_vec = company_shap[shap_features].astype(float)

            # Get top 15 features by absolute SHAP for this company
            abs_shap = shap_vec.abs().sort_values(ascending=False)
            top_features = abs_shap.head(15).index.tolist()

            waterfall_data = pd.DataFrame({
                'feature': top_features,
                'shap_value': [shap_vec[f] for f in top_features],
            })
            waterfall_data['direction'] = waterfall_data['shap_value'].apply(
                lambda x: 'Pushes toward GREENWASHING' if x > 0 else 'Pushes toward GENUINE'
            )
            waterfall_data['abs_value'] = waterfall_data['shap_value'].abs()
            waterfall_data['is_new'] = waterfall_data['feature'].isin(new_feature_set)
            waterfall_data = waterfall_data.sort_values('shap_value', ascending=True)

            fig = go.Figure(go.Bar(
                x=waterfall_data['shap_value'],
                y=waterfall_data['feature'],
                orientation='h',
                marker=dict(
                    color=waterfall_data['shap_value'],
                    colorscale='RdBu_r',
                    cmid=0,
                    showscale=True,
                    colorbar=dict(title='SHAP value'),
                ),
                text=[f"{v:+.4f}" for v in waterfall_data['shap_value']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>SHAP: %{x:+.4f}<br>%{customdata}<extra></extra>',
                customdata=waterfall_data['direction'],
            ))
            fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
            fig.update_layout(
                title=f'Top 15 SHAP Contributions for {selected}',
                xaxis_title='SHAP value (red = greenwashing, blue = genuine)',
                yaxis_title='Feature',
                height=600,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Verbal interpretation
            pos_count = (waterfall_data['shap_value'] > 0).sum()
            neg_count = (waterfall_data['shap_value'] < 0).sum()
            net_shap = float(shap_vec.sum())
            total_pos = float(shap_vec[shap_vec > 0].sum())
            total_neg = float(shap_vec[shap_vec < 0].sum())

            col1, col2, col3 = st.columns(3)
            col1.metric("Net SHAP (sum)", f"{net_shap:+.3f}",
                        help="Sum of all SHAP values. Positive = model leans greenwashing.")
            col2.metric("Pushing toward GW", f"{total_pos:+.3f}",
                        help="Total of positive SHAP contributions.")
            col3.metric("Pushing toward GENUINE", f"{total_neg:+.3f}",
                        help="Total of negative SHAP contributions.")

            new_in_top = waterfall_data[waterfall_data['is_new']]['feature'].tolist()
            if new_in_top:
                st.info(
                    f"**{len(new_in_top)} of the top 15 contributing features are NEW (Categories 7-10):**  "
                    + ", ".join(new_in_top)
                )
            else:
                st.warning("None of the top contributing features are from the new categories. "
                           "This company is being flagged primarily by original features.")

    # ====================================================================
    # SECTION 4: FEATURE DISTRIBUTION EXPLORER (linked to SHAP)
    # ====================================================================
    st.markdown("---")
    st.subheader("4. SHAP-Ranked Feature Distribution")
    st.markdown("Explore the distribution of features ranked by their SHAP impact.")

    if not fm.empty:
        numeric_cols = fm.select_dtypes(include=[np.number]).columns.tolist()
        important = shap_fi.head(40)['feature'].tolist()
        available = [f for f in important if f in numeric_cols]

        if available:
            selected_feature = st.selectbox(
                "Choose a feature to explore (sorted by SHAP impact)",
                available,
                key='shap_feature_explorer',
            )
            if selected_feature:
                shap_rank_idx = shap_fi[shap_fi['feature'] == selected_feature].index
                shap_rank = (int(shap_rank_idx[0]) + 1) if len(shap_rank_idx) > 0 else None
                shap_val = float(shap_fi[shap_fi['feature'] == selected_feature]['shap_importance'].iloc[0]) if shap_rank else 0
                is_new = selected_feature in new_feature_set
                badge = " *(NEW Cat 7-10)*" if is_new else ""
                st.markdown(f"**SHAP Rank: #{shap_rank}** | Mean |SHAP|: {shap_val:.6f}{badge}")

                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(
                        fm, x=selected_feature, nbins=30,
                        title=f'Distribution of {selected_feature}',
                        color_discrete_sequence=['#3498db'],
                        marginal='box',
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    if 'sector' in fm.columns:
                        fig = px.box(
                            fm, x='sector', y=selected_feature,
                            title=f'{selected_feature} by Sector',
                            color='sector',
                        )
                        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)

    # ====================================================================
    # SECTION 5: SHAP CORRELATION HEATMAP (TOP FEATURES)
    # ====================================================================
    st.markdown("---")
    st.subheader("5. Top SHAP Features Correlation")

    if not fm.empty and not shap_fi.empty:
        top_features = shap_fi.head(10)['feature'].tolist()
        top_features = [f for f in top_features if f in fm.columns]

        if len(top_features) >= 2:
            corr = fm[top_features].corr()
            fig = px.imshow(
                corr, text_auto='.2f',
                title='Correlation Heatmap (Top 10 SHAP Features)',
                color_continuous_scale='RdBu_r',
                aspect='auto',
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    # ====================================================================
    # SECTION 6: LEGACY TEXT EXPLANATIONS (collapsed by default)
    # ====================================================================
    with st.expander("Legacy SHAP text explanations (from earlier model)", expanded=False):
        st.code(data.get('shap_explanations', 'Not available'), language=None)


# ============================================================================
# PAGE 13: HYBRID XGBOOST-CATBOOST MODEL
# ============================================================================

def page_hybrid_model(data):
    """Hybrid XGBoost-CatBoost model with Monte Carlo uncertainty estimation.

    Inspired by:
        Optimized hybrid XGBoost-CatBoost model for enhanced prediction
        using Monte Carlo simulations (ScienceDirect)

    Adapted for ESG greenwashing detection.
    """

    st.title("Hybrid XGBoost-CatBoost Model with Monte Carlo Uncertainty")
    st.markdown(
        "**Inspired by recent research** on hybrid gradient-boosting ensembles. This model "
        "blends XGBoost (great at numerical interactions) with CatBoost (robust to overfitting "
        "and categorical features), then runs Monte Carlo simulation to quantify prediction "
        "uncertainty per company."
    )

    hybrid_pred = data.get('hybrid_predictions', pd.DataFrame())
    hybrid_metrics = data.get('hybrid_metrics', {})
    hybrid_fi = data.get('hybrid_feature_importance', pd.DataFrame())
    hybrid_shap_fi = data.get('hybrid_shap_importance', pd.DataFrame())
    hybrid_shap_all = data.get('hybrid_shap_values_all', pd.DataFrame())

    if hybrid_pred.empty or not hybrid_metrics:
        st.error(
            "**Hybrid model outputs not found.** Please run `python hybrid_xgb_catboost.py` "
            "from the project root to train the hybrid model."
        )
        st.info("Required files:\n"
                "- `data/processed/hybrid_predictions.csv`\n"
                "- `data/processed/hybrid_model_metrics.json`\n"
                "- `data/processed/hybrid_feature_importance.csv`")
        return

    # ====================================================================
    # SECTION 1: MODEL CONFIGURATION & METRICS
    # ====================================================================
    st.subheader("1. Model Configuration")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Features Used", f"{hybrid_metrics.get('n_features', 0)}")
    col2.metric("Training Samples", f"{hybrid_metrics.get('train_size', 0)}")
    col3.metric("Validation Samples", f"{hybrid_metrics.get('val_size', 0)}")
    col4.metric("Test Samples", f"{hybrid_metrics.get('test_size', 0)}")

    col_a, col_b, col_c = st.columns(3)
    alpha_xgb = hybrid_metrics.get('optimal_alpha_xgboost', 0.5)
    alpha_cat = hybrid_metrics.get('optimal_alpha_catboost', 0.5)
    val_loss = hybrid_metrics.get('validation_logloss', 0)
    col_a.metric("XGBoost Weight (α)", f"{alpha_xgb:.3f}",
                 help="Optimal weight on XGBoost found by minimizing log-loss")
    col_b.metric("CatBoost Weight (1−α)", f"{alpha_cat:.3f}",
                 help="Optimal weight on CatBoost")
    col_c.metric("Validation Log-Loss", f"{val_loss:.4f}",
                 help="Lower is better. Achieved by optimal blend.")

    if alpha_xgb < 0.1:
        st.info(
            f"**Finding:** The optimization picked α = {alpha_xgb:.3f}, meaning **CatBoost decisively wins** "
            f"on the validation set for this dataset. The hybrid effectively becomes pure CatBoost. "
            f"This is itself a finding — for ESG features, CatBoost's L2 regularization handles "
            f"sparse policy/temporal features better than XGBoost."
        )
    elif alpha_xgb > 0.9:
        st.info(
            f"**Finding:** Optimization picked α = {alpha_xgb:.3f}, meaning XGBoost dominates. "
            f"The hybrid is effectively pure XGBoost."
        )
    else:
        st.success(
            f"**True hybrid:** Both models contribute meaningfully. "
            f"XGBoost: {alpha_xgb:.0%}, CatBoost: {alpha_cat:.0%}."
        )

    st.markdown("---")

    # ====================================================================
    # SECTION 2: MODEL COMPARISON
    # ====================================================================
    st.subheader("2. Model Performance Comparison")

    metrics_data = hybrid_metrics.get('metrics', {})
    if metrics_data:
        rows = []
        for key, m in metrics_data.items():
            rows.append({
                'Model': m['model'],
                'Accuracy': round(m['accuracy'], 4),
                'Precision': round(m['precision'], 4),
                'Recall': round(m['recall'], 4),
                'F1 Score': round(m['f1'], 4),
                'ROC-AUC': round(m['roc_auc'], 4),
            })
        comp_df = pd.DataFrame(rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Bar chart comparison
        long_df = comp_df.melt(
            id_vars='Model',
            value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
            var_name='Metric',
            value_name='Score',
        )
        fig = px.bar(
            long_df, x='Metric', y='Score', color='Model', barmode='group',
            title='Model Performance Comparison',
            color_discrete_sequence=['#3498db', '#9b59b6', '#27ae60'],
            text='Score',
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(yaxis_range=[0, 1.05], height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ====================================================================
    # SECTION 3: MONTE CARLO UNCERTAINTY ANALYSIS
    # ====================================================================
    st.subheader("3. Monte Carlo Uncertainty Analysis")
    st.markdown(
        f"**{hybrid_metrics.get('monte_carlo_runs', 20)} bootstrap runs** with "
        f"{hybrid_metrics.get('monte_carlo_sample_frac', 0.8) * 100:.0f}% sampling. "
        "Each company's prediction uncertainty (standard deviation) reveals how stable the "
        "model's decision is. High uncertainty means the prediction would flip easily with "
        "different training data — critical for risk-based decisions."
    )

    # Distribution of uncertainty
    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.histogram(
            hybrid_pred, x='uncertainty', nbins=40,
            title='Prediction Uncertainty Distribution',
            labels={'uncertainty': 'Standard deviation (MC)'},
            color_discrete_sequence=['#e67e22'],
            marginal='box',
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Scatter: prob vs uncertainty
        fig = px.scatter(
            hybrid_pred, x='hybrid_prob', y='uncertainty',
            color='predicted_class',
            title='Prediction vs Uncertainty (per company)',
            labels={'hybrid_prob': 'Hybrid probability', 'uncertainty': 'MC Std (uncertainty)'},
            color_continuous_scale='RdBu_r',
            hover_data=['company_name'],
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # High-uncertainty companies table
    st.markdown("**Top 15 Highest-Uncertainty Companies (decision boundary cases):**")
    st.markdown("These companies sit very close to the GW/genuine boundary — small "
                "changes in training data flip the prediction. Investigate these manually.")

    high_unc = hybrid_pred.nlargest(15, 'uncertainty')[
        ['company_name', 'sector', 'hybrid_prob', 'hybrid_mc_mean',
         'uncertainty', 'hybrid_ci_low', 'hybrid_ci_high', 'predicted_class']
    ].copy()
    high_unc['hybrid_prob'] = high_unc['hybrid_prob'].round(4)
    high_unc['hybrid_mc_mean'] = high_unc['hybrid_mc_mean'].round(4)
    high_unc['uncertainty'] = high_unc['uncertainty'].round(4)
    high_unc['hybrid_ci_low'] = high_unc['hybrid_ci_low'].round(4)
    high_unc['hybrid_ci_high'] = high_unc['hybrid_ci_high'].round(4)
    high_unc['CI Width'] = (high_unc['hybrid_ci_high'] - high_unc['hybrid_ci_low']).round(4)
    st.dataframe(high_unc, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ====================================================================
    # SECTION 4: PER-COMPANY HYBRID PREDICTION EXPLORER
    # ====================================================================
    st.subheader("4. Per-Company Hybrid Prediction Explorer")

    company_list = sorted(hybrid_pred['company_name'].dropna().unique().tolist())
    selected = st.selectbox(
        "Select a company",
        company_list,
        key='hybrid_company_selector',
    )

    if selected:
        row = hybrid_pred[hybrid_pred['company_name'] == selected].iloc[0]

        col_w, col_x, col_y, col_z = st.columns(4)
        col_w.metric("XGBoost prob", f"{row['xgboost_prob']:.4f}")
        col_x.metric("CatBoost prob", f"{row['catboost_prob']:.4f}")
        col_y.metric("Hybrid prob", f"{row['hybrid_prob']:.4f}")
        col_z.metric("Uncertainty (±)", f"{row['uncertainty']:.4f}")

        # Confidence interval visualization
        fig = go.Figure()

        # CI bar
        fig.add_trace(go.Scatter(
            x=[row['hybrid_ci_low'], row['hybrid_ci_high']],
            y=['Hybrid CI', 'Hybrid CI'],
            mode='lines',
            line=dict(color='#3498db', width=20),
            name='95% Confidence Interval',
            showlegend=False,
        ))

        # Mean prediction marker
        fig.add_trace(go.Scatter(
            x=[row['hybrid_mc_mean']],
            y=['Hybrid CI'],
            mode='markers',
            marker=dict(color='#e74c3c', size=18, symbol='diamond'),
            name='MC Mean',
            text=[f"{row['hybrid_mc_mean']:.3f}"],
            textposition='top center',
        ))

        # Decision threshold line
        fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                      annotation_text="Decision threshold (0.5)",
                      annotation_position="top")

        fig.update_layout(
            title=f'Prediction Confidence Interval for {selected}',
            xaxis_title='Greenwashing probability',
            xaxis_range=[0, 1],
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Verbal interpretation
        prob = row['hybrid_prob']
        unc = row['uncertainty']
        ci_low = row['hybrid_ci_low']
        ci_high = row['hybrid_ci_high']

        if unc < 0.05:
            st.success(f"**Highly confident prediction.** Standard deviation = {unc:.3f}, "
                       f"CI = [{ci_low:.3f}, {ci_high:.3f}]. The prediction is stable.")
        elif unc < 0.15:
            st.info(f"**Moderately confident prediction.** Std = {unc:.3f}, "
                    f"CI = [{ci_low:.3f}, {ci_high:.3f}]. Some sensitivity to training data.")
        else:
            st.warning(f"**Low-confidence prediction.** Std = {unc:.3f}, "
                       f"CI = [{ci_low:.3f}, {ci_high:.3f}]. The prediction is unstable — "
                       f"this company sits near the decision boundary. Manual review recommended.")

    st.markdown("---")

    # ====================================================================
    # SECTION 5: HYBRID FEATURE IMPORTANCE
    # ====================================================================
    st.subheader("5. Hybrid Feature Importance (XGB + CatBoost)")
    st.markdown(
        "Feature importance averaged from both base models, weighted by the optimal blend α."
    )

    if not hybrid_fi.empty:
        new_feature_set = {
            'paris_agreement_alignment', 'eu_taxonomy_alignment', 'sec_climate_alignment',
            'tcfd_alignment', 'sdg_alignment', 'gri_standards_alignment',
            'regulatory_breadth_index', 'total_policy_density', 'policy_specificity_score',
            'policy_esg_gap', 'framework_consistency_score', 'regulatory_readiness_score',
            'promotional_intent_density', 'defensive_intent_density', 'factual_intent_density',
            'strategic_intent_density', 'narrative_credibility_index', 'promotional_dominance_score',
            'intent_diversity_score', 'defensive_to_factual_ratio', 'sentiment_intent_divergence',
            'news_greenwashing_signal', 'past_achievement_density', 'present_action_density',
            'specific_future_density', 'vague_future_density', 'temporal_balance_score',
            'commitment_credibility_score', 'temporal_specificity_ratio', 'progress_to_promise_ratio',
            'year_mention_density', 'temporal_greenwashing_signal',
            'multi_signal_greenwashing_score', 'esg_linguistic_credibility_index',
            'aggregate_esg_nlp_score',
        }

        top20 = hybrid_fi.head(20).copy()
        top20['is_new'] = top20['feature'].isin(new_feature_set)
        top20['color'] = top20['is_new'].map({True: '#27ae60', False: '#e74c3c'})
        top20 = top20.sort_values('hybrid_importance', ascending=True)

        fig = go.Figure(go.Bar(
            x=top20['hybrid_importance'],
            y=top20['feature'],
            orientation='h',
            marker=dict(color=top20['color'].tolist()),
            text=top20['hybrid_importance'].round(4),
            textposition='outside',
        ))
        fig.update_layout(
            title='Top 20 Features by Hybrid Model Importance (green = NEW Cat 7-10)',
            xaxis_title='Hybrid Importance (normalized)',
            yaxis_title='Feature',
            height=600,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ====================================================================
    # SECTION 6: HYBRID SHAP EXPLANATIONS
    # ====================================================================
    st.subheader("6. Hybrid Model SHAP Explanations")
    st.markdown(
        "SHAP values computed on the dominant base model (CatBoost given α ≈ 0). "
        "These explain WHY the hybrid model assigns each company its probability."
    )

    if not hybrid_shap_fi.empty and not hybrid_shap_all.empty:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Top 15 SHAP features (Hybrid model):**")
            top15_shap = hybrid_shap_fi.head(15).copy()
            top15_shap = top15_shap.sort_values('shap_importance', ascending=True)
            fig = go.Figure(go.Bar(
                x=top15_shap['shap_importance'],
                y=top15_shap['feature'],
                orientation='h',
                marker_color='#9b59b6',
            ))
            fig.update_layout(
                xaxis_title='Mean |SHAP|',
                height=500,
                margin=dict(l=200),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Per-company SHAP waterfall (top 12 features):**")
            sel_for_shap = st.selectbox(
                "Company",
                company_list,
                index=company_list.index(selected) if selected in company_list else 0,
                key='hybrid_shap_selector',
            )

            if sel_for_shap and sel_for_shap in hybrid_shap_all['company_name'].values:
                comp_shap = hybrid_shap_all[hybrid_shap_all['company_name'] == sel_for_shap].iloc[0]
                shap_features = [c for c in hybrid_shap_all.columns if c != 'company_name']
                shap_vec = comp_shap[shap_features].astype(float)
                top12_idx = shap_vec.abs().nlargest(12).index
                wf = pd.DataFrame({
                    'feature': top12_idx,
                    'shap': [shap_vec[f] for f in top12_idx],
                }).sort_values('shap', ascending=True)

                fig = go.Figure(go.Bar(
                    x=wf['shap'],
                    y=wf['feature'],
                    orientation='h',
                    marker=dict(
                        color=wf['shap'],
                        colorscale='RdBu_r',
                        cmid=0,
                    ),
                    text=[f"{v:+.3f}" for v in wf['shap']],
                    textposition='outside',
                ))
                fig.add_vline(x=0, line_dash='dash', line_color='black')
                fig.update_layout(
                    xaxis_title='SHAP value',
                    height=500,
                    margin=dict(l=200),
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Hybrid SHAP files not found. Run the SHAP computation step in `hybrid_xgb_catboost.py`.")

    # ====================================================================
    # SECTION 7: METHOD EXPLANATION
    # ====================================================================
    with st.expander("How the Hybrid Model Works (click to expand)", expanded=False):
        st.markdown("""
        ### Hybrid XGBoost-CatBoost with Monte Carlo

        **Training pipeline:**
        1. Split data 60/20/20 (train/validation/test)
        2. Train XGBoost classifier on training set with early stopping on validation
        3. Train CatBoost classifier on training set with early stopping on validation
        4. Find optimal blend weight α by minimizing log-loss on validation:
           ```
           p_hybrid = α × p_xgboost + (1 − α) × p_catboost
           ```
           Grid search over α ∈ [0, 1] in steps of 0.01

        **Monte Carlo uncertainty (20 runs):**
        ```
        for run in 1..20:
            X_boot = bootstrap_sample(X_train, frac=0.8, seed=run)
            xgb_run = train_xgboost(X_boot)
            cat_run = train_catboost(X_boot)
            p_run = α × xgb_run.predict(X_all) + (1−α) × cat_run.predict(X_all)
            mc_predictions.append(p_run)

        mc_mean = mean(mc_predictions, axis=0)
        mc_std  = std(mc_predictions, axis=0)
        ci_low  = percentile(mc_predictions, 2.5)
        ci_high = percentile(mc_predictions, 97.5)
        ```

        **Why this design:**
        - **XGBoost** is great at capturing complex numerical interactions
        - **CatBoost** has stronger regularization (L2 leaf reg) and handles sparse features well
        - Their decision boundaries differ — blending captures complementary signals
        - **Monte Carlo** gives uncertainty estimates — critical for risk decisions
          (a high-probability prediction with high uncertainty should NOT be acted on)

        **Inspired by:** Optimized hybrid XGBoost-CatBoost model for enhanced prediction
        of concrete strength and reliability analysis using Monte Carlo simulations
        (ScienceDirect)
        """)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit application entry point."""

    data = load_data()
    page, filters = render_sidebar(data)

    if page == "Risk Score Dashboard":
        page_risk_dashboard(data, filters)
    elif page == "Model Performance":
        page_model_performance(data)
    elif page == "Feature Importance":
        page_feature_importance(data)
    elif page == "Company Deep Dive":
        page_company_deep_dive(data)
    elif page == "Company Search & Analysis":
        page_company_search(data)
    elif page == "Real-Time Intelligence":
        page_realtime_intelligence(data)
    elif page == "ESG Report Analyzer (AI)":
        page_esg_report_analyzer(data)
    elif page == "Report Generator":
        page_report_generator(data)
    elif page == "Advanced Explainability":
        page_advanced_explainability(data)
    elif page == "Company Comparison":
        page_company_comparison(data)
    elif page == "Time-Series Risk Tracking":
        page_timeseries_tracking(data)
    elif page == "SHAP Explanations":
        page_shap_explanations(data)
    elif page == "Hybrid XGBoost-CatBoost Model":
        page_hybrid_model(data)


if __name__ == "__main__":
    main()
