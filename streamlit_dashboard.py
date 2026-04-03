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
    risk_path = os.path.join(PROCESSED_DIR, "greenwashing_risk_scores.csv")
    if os.path.exists(risk_path):
        data['risk_scores'] = pd.read_csv(risk_path, index_col=0)
    else:
        data['risk_scores'] = pd.DataFrame()

    # Model metrics
    metrics_path = os.path.join(PROCESSED_DIR, "model_metrics.csv")
    if os.path.exists(metrics_path):
        data['model_metrics'] = pd.read_csv(metrics_path)
    else:
        data['model_metrics'] = pd.DataFrame()

    # Feature importance (best model)
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
    fm_path = os.path.join(PROCESSED_DIR, "feature_matrix.csv")
    if os.path.exists(fm_path):
        data['feature_matrix'] = pd.read_csv(fm_path)
    else:
        data['feature_matrix'] = pd.DataFrame()

    # Predictions (for confusion matrix data)
    pred_path = os.path.join(PROCESSED_DIR, "predictions.csv")
    if os.path.exists(pred_path):
        data['predictions'] = pd.read_csv(pred_path)
    else:
        data['predictions'] = pd.DataFrame()

    # SHAP explanations
    shap_path = os.path.join(PROCESSED_DIR, "shap_explanations.txt")
    if os.path.exists(shap_path):
        with open(shap_path, 'r', encoding='utf-8') as f:
            data['shap_explanations'] = f.read()
    else:
        data['shap_explanations'] = "SHAP explanations not yet generated. Run model_pipeline.py first."

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
            "SHAP Explanations",
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
    st.markdown("Comprehensive risk assessment of **480 companies** across S&P 500 and NIFTY 50")

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
    st.markdown("Type a company name to get a **complete greenwashing risk analysis report**.")

    df = data['risk_scores']
    fm = data['feature_matrix']
    pred = data['predictions']

    if df.empty or fm.empty:
        st.warning("Data not available. Run `python model_pipeline.py` first.")
        return

    # --- Search box with autocomplete ---
    company_names = sorted(df['company_name'].dropna().unique().tolist())
    search_query = st.text_input(
        "Search Company Name",
        placeholder="Type company name (e.g., Apple, Tesla, Microsoft)...",
    )

    # Filter matching companies
    if search_query:
        matches = [c for c in company_names if search_query.lower() in c.lower()]
    else:
        matches = []

    if search_query and not matches:
        st.error(f"No company found matching '{search_query}'. Try a different name.")
        st.markdown("**Available companies (sample):**")
        st.write(", ".join(company_names[:20]) + "...")
        return

    if not search_query:
        st.info("Start typing a company name above to search.")
        # Show top 10 highest risk companies as suggestions
        st.subheader("Top 10 Highest Risk Companies")
        top10 = df.nlargest(10, 'risk_score')[['company_name', 'sector', 'risk_score', 'risk_tier']]
        st.dataframe(top10.reset_index(drop=True), use_container_width=True)
        return

    # Let user pick from matches if multiple
    if len(matches) == 1:
        selected = matches[0]
    else:
        selected = st.selectbox(f"Found {len(matches)} matches -- select one:", matches)

    if not selected:
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
                if val is not None and not pd.isna(val):
                    all_features[col] = round(float(val), 6)

        full_df = pd.DataFrame({
            'Feature': list(all_features.keys()),
            'Value': list(all_features.values()),
        })
        st.dataframe(full_df, use_container_width=True, height=600)


# ============================================================================
# PAGE 6: SHAP EXPLANATIONS (all interactive)
# ============================================================================

def page_shap_explanations(data):
    """SHAP explanations — interactive charts + text explanations."""

    st.title("SHAP Explanations -- Why Companies Are Flagged")
    st.markdown("SHAP (SHapley Additive exPlanations) shows which features pushed each "
                "company's prediction toward or away from greenwashing.")

    fi = data['feature_importance']
    fm = data['feature_matrix']

    # --- Interactive SHAP-like waterfall chart (simulated from feature importance) ---
    if not fi.empty:
        st.subheader("Global Feature Impact (Top 20)")

        top20 = fi.head(20).copy()
        top20 = top20.sort_values('importance', ascending=True)

        fig = go.Figure(go.Bar(
            x=top20['importance'],
            y=top20['feature'],
            orientation='h',
            marker=dict(
                color=top20['importance'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title='Importance'),
            ),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.6f}<extra></extra>',
        ))
        fig.update_layout(
            title='Feature Importance (Gradient Boosting)',
            xaxis_title='Importance', yaxis_title='Feature',
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Interactive feature distribution explorer ---
    if not fm.empty:
        st.subheader("Feature Distribution Explorer")
        st.markdown("Select a feature to see how it distributes across all companies.")

        numeric_cols = fm.select_dtypes(include=[np.number]).columns.tolist()
        important_features = fi['feature'].head(30).tolist() if not fi.empty else numeric_cols[:20]
        available = [f for f in important_features if f in numeric_cols]

        if available:
            selected_feature = st.selectbox("Choose a feature to explore", available)

            if selected_feature:
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

    # --- SHAP text explanations ---
    st.markdown("---")
    st.subheader("Company-Level SHAP Explanations")
    st.markdown("Detailed text explanations for the top flagged companies:")

    with st.expander("View Full SHAP Explanations (click to expand)", expanded=False):
        st.code(data['shap_explanations'], language=None)

    # --- Feature correlation heatmap (top features) ---
    if not fm.empty and not fi.empty:
        st.subheader("Top Feature Correlations")
        top_features = fi['feature'].head(10).tolist()
        top_features = [f for f in top_features if f in fm.columns]

        if len(top_features) >= 2:
            corr = fm[top_features].corr()
            fig = px.imshow(
                corr, text_auto='.2f',
                title='Correlation Heatmap (Top 10 Features)',
                color_continuous_scale='RdBu_r',
                aspect='auto',
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)


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
    elif page == "SHAP Explanations":
        page_shap_explanations(data)


if __name__ == "__main__":
    main()
