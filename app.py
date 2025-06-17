import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import re
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib
matplotlib.use('Agg')

# Try to import statsmodels, install if not available
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:
    st.warning("statsmodels not installed. Some advanced statistical features may be limited.")
    sm = None

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="🤖 Smart Data Analysis Chat",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    .prompt-suggestion {
        background-color: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        display: inline-block;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .prompt-suggestion:hover {
        background-color: #667eea;
        color: white;
    }
    
    .filter-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configure Gemini API
@st.cache_resource
def initialize_gemini():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found. Please set it in your .env file.")
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

# Initialize session state
def initialize_session_state():
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'data_info' not in st.session_state:
        st.session_state.data_info = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processed_insights' not in st.session_state:
        st.session_state.processed_insights = None
    if 'visualization_history' not in st.session_state:
        st.session_state.visualization_history = []
    if 'analysis_reports' not in st.session_state:
        st.session_state.analysis_reports = []
    if 'chart_counter' not in st.session_state:
        st.session_state.chart_counter = 0

def detect_outliers(df, column):
    """Detect outliers using multiple methods"""
    if column not in df.columns or df[column].dtype not in ['int64', 'float64']:
        return {}
    
    data = df[column].dropna()
    
    # IQR method
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    # Z-score method
    z_scores = np.abs(stats.zscore(data))
    z_outliers = data[z_scores > 3]
    
    # Modified Z-score method
    median = data.median()
    mad = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad if mad > 0 else np.zeros_like(data)
    modified_z_outliers = data[np.abs(modified_z_scores) > 3.5]
    
    return {
        'iqr_method': {
            'count': len(iqr_outliers),
            'percentage': len(iqr_outliers) / len(data) * 100,
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'outliers': iqr_outliers.tolist()[:10]  # Limit to first 10
        },
        'z_score_method': {
            'count': len(z_outliers),
            'percentage': len(z_outliers) / len(data) * 100,
            'outliers': z_outliers.tolist()[:10]
        },
        'modified_z_score': {
            'count': len(modified_z_outliers),
            'percentage': len(modified_z_outliers) / len(data) * 100,
            'outliers': modified_z_outliers.tolist()[:10]
        }
    }

def perform_advanced_analysis(df):
    """Perform advanced statistical analysis"""
    results = {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Normality tests
    normality_results = {}
    for col in numeric_cols[:10]:  # Limit to first 10 columns
        data = df[col].dropna()
        if len(data) > 3:
            try:
                statistic, p_value = stats.shapiro(data) if len(data) < 5000 else stats.normaltest(data)
                normality_results[col] = {
                    'is_normal': bool(p_value > 0.05),  # Convert to bool
                    'p_value': float(p_value),  # Convert to float
                    'interpretation': 'Normally distributed' if p_value > 0.05 else 'Not normally distributed'
                }
            except:
                pass
    
    results['normality_tests'] = normality_results
    
    # Trend analysis for numeric columns
    trend_analysis = {}
    for col in numeric_cols[:5]:  # Limit to first 5 columns
        data = df[col].dropna()
        if len(data) > 10:
            try:
                x = np.arange(len(data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
                trend_analysis[col] = {
                    'trend': 'Increasing' if slope > 0 else 'Decreasing',
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05)
                }
            except:
                pass
    
    results['trend_analysis'] = trend_analysis
    
    # Variance analysis
    if len(numeric_cols) > 1:
        high_variance_cols = []
        low_variance_cols = []
        
        for col in numeric_cols:
            try:
                mean_val = df[col].mean()
                if mean_val != 0:
                    cv = df[col].std() / mean_val
                    if cv > 1:
                        high_variance_cols.append((col, float(cv)))
                    elif cv < 0.1:
                        low_variance_cols.append((col, float(cv)))
            except:
                pass
        
        results['variance_analysis'] = {
            'high_variance': high_variance_cols[:5],
            'low_variance': low_variance_cols[:5]
        }
    
    return results

def comprehensive_data_analysis(df):
    """Comprehensive data structure analysis with detailed statistics"""
    
    # Basic info
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(include=['object', 'category', 'bool']).columns)
    datetime_cols = list(df.select_dtypes(include=['datetime64']).columns)
    
    # Try to detect date columns that might be stored as strings
    for col in categorical_cols[:]:  # Use slice to avoid modifying list during iteration
        if df[col].dtype == 'object':
            try:
                # Check if it could be a date
                sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if sample and isinstance(sample, str):
                    if re.match(r'\d{4}-\d{2}-\d{2}', sample) or re.match(r'\d{2}/\d{2}/\d{4}', sample):
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        datetime_cols.append(col)
                        categorical_cols.remove(col)
            except:
                pass
    
    # Detailed statistical analysis
    stats_summary = {}
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            stats_summary[col] = {
                'count': int(len(col_data)),
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'mode': float(col_data.mode().iloc[0]) if len(col_data.mode()) > 0 else None,
                'std': float(col_data.std()),
                'variance': float(col_data.var()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'range': float(col_data.max() - col_data.min()),
                'q1': float(col_data.quantile(0.25)),
                'q3': float(col_data.quantile(0.75)),
                'iqr': float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                'skewness': float(col_data.skew()),
                'kurtosis': float(col_data.kurtosis()),
                'zero_values': int((col_data == 0).sum()),
                'negative_values': int((col_data < 0).sum()),
                'positive_values': int((col_data > 0).sum()),
                'outliers_iqr': int(len(col_data[(col_data < (col_data.quantile(0.25) - 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25)))) | 
                                                    (col_data > (col_data.quantile(0.75) + 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25))))])),
                'percentiles': {
                    '10th': float(col_data.quantile(0.1)),
                    '25th': float(col_data.quantile(0.25)),
                    '50th': float(col_data.quantile(0.5)),
                    '75th': float(col_data.quantile(0.75)),
                    '90th': float(col_data.quantile(0.9)),
                    '95th': float(col_data.quantile(0.95)),
                    '99th': float(col_data.quantile(0.99))
                }
            }
    
    # Categorical analysis
    categorical_summary = {}
    for col in categorical_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            value_counts = col_data.value_counts()
            categorical_summary[col] = {
                'unique_count': int(col_data.nunique()),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'least_frequent': str(value_counts.index[-1]) if len(value_counts) > 0 else None,
                'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                'top_5_values': {str(k): int(v) for k, v in value_counts.head().to_dict().items()},
                'distribution_evenness': float(value_counts.std() / value_counts.mean()) if value_counts.mean() > 0 else 0
            }
    
    # Date/Time analysis
    datetime_summary = {}
    for col in datetime_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            datetime_summary[col] = {
                'min_date': str(col_data.min()),
                'max_date': str(col_data.max()),
                'date_range_days': int((col_data.max() - col_data.min()).days),
                'unique_dates': int(col_data.nunique()),
                'most_frequent_date': str(col_data.mode().iloc[0]) if len(col_data.mode()) > 0 else None,
                'weekend_count': int(col_data[col_data.dt.dayofweek.isin([5, 6])].count()),
                'weekday_count': int(col_data[~col_data.dt.dayofweek.isin([5, 6])].count())
            }
    
    # Correlation analysis
    correlation_data = {}
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        correlation_data = {
            'matrix': corr_matrix.to_dict(),
            'strong_positive': [],
            'strong_negative': [],
            'moderate_correlations': []
        }
        
        # Find significant correlations
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                
                if corr_val > 0.7:
                    correlation_data['strong_positive'].append({
                        'variables': f"{col1} vs {col2}",
                        'correlation': float(corr_val)
                    })
                elif corr_val < -0.7:
                    correlation_data['strong_negative'].append({
                        'variables': f"{col1} vs {col2}",
                        'correlation': float(corr_val)
                    })
                elif abs(corr_val) > 0.3:
                    correlation_data['moderate_correlations'].append({
                        'variables': f"{col1} vs {col2}",
                        'correlation': float(corr_val)
                    })
    
    # Data quality assessment
    data_quality = {
        'completeness_score': float(100 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)),
        'duplicate_rows': int(df.duplicated().sum()),
        'duplicate_percentage': float(df.duplicated().sum() / len(df) * 100),
        'columns_with_nulls': [col for col in df.columns if df[col].isnull().sum() > 0],
        'high_null_columns': [col for col in df.columns if df[col].isnull().sum() / len(df) > 0.5],
        'constant_columns': [col for col in df.columns if df[col].nunique() <= 1],
        'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024)
    }
    
    # Sample data analysis
    sample_data = df.head(10).to_dict('records')
    
    # Advanced analysis
    advanced_stats = perform_advanced_analysis(df)
    
    return {
        'basic_info': {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.astype(str).to_dict(),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': datetime_cols,
            'total_cells': int(df.shape[0] * df.shape[1]),
            'non_null_cells': int(df.count().sum())
        },
        'statistical_summary': stats_summary,
        'categorical_summary': categorical_summary,
        'datetime_summary': datetime_summary,
        'correlation_analysis': correlation_data,
        'data_quality': data_quality,
        'sample_data': sample_data,
        'missing_values': {col: int(df[col].isnull().sum()) for col in df.columns},
        'unique_counts': {col: int(df[col].nunique()) for col in df.columns},
        'advanced_analysis': advanced_stats
    }

def get_unique_key():
    """Generate unique key for streamlit elements"""
    st.session_state.chart_counter += 1
    return f"chart_{st.session_state.chart_counter}_{datetime.now().timestamp()}"

def create_comprehensive_visualizations(df, chart_type, x_col=None, y_col=None, title="", color_col=None, **kwargs):
    """Create comprehensive visualizations with multiple chart types"""
    
    try:
        if chart_type == 'overview_dashboard':
            # Create a comprehensive dashboard
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
            
            if len(numeric_cols) >= 2:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Correlation Heatmap', 'Distribution Plot', 'Box Plots', 'Trend Analysis'),
                    specs=[[{"type": "heatmap"}, {"type": "histogram"}],
                           [{"type": "box"}, {"type": "scatter"}]]
                )
                
                # Correlation heatmap
                corr_matrix = df[numeric_cols].corr()
                fig.add_trace(
                    go.Heatmap(z=corr_matrix.values, 
                              x=corr_matrix.columns, 
                              y=corr_matrix.columns,
                              colorscale='RdBu',
                              text=corr_matrix.round(2).values,
                              texttemplate="%{text}",
                              textfont={"size": 10}),
                    row=1, col=1
                )
                
                # Distribution
                fig.add_trace(
                    go.Histogram(x=df[numeric_cols[0]], name=numeric_cols[0], opacity=0.7),
                    row=1, col=2
                )
                
                # Box plots
                for i, col in enumerate(numeric_cols[:2]):
                    fig.add_trace(
                        go.Box(y=df[col], name=col),
                        row=2, col=1
                    )
                
                # Scatter plot
                if len(numeric_cols) >= 2:
                    fig.add_trace(
                        go.Scatter(x=df[numeric_cols[0]], y=df[numeric_cols[1]], 
                                  mode='markers', name=f'{numeric_cols[0]} vs {numeric_cols[1]}',
                                  opacity=0.6),
                        row=2, col=2
                    )
                
                fig.update_layout(height=800, title_text="Data Overview Dashboard", showlegend=True)
                return fig
        
        elif chart_type == 'correlation_heatmap':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                
                # Create enhanced heatmap with annotations
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.round(3).values,
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title=f"Correlation Matrix - {title}",
                    width=700,
                    height=600
                )
                return fig
        
        elif chart_type == 'distribution_analysis':
            if y_col and y_col in df.columns:
                # Create subplot with histogram and box plot
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(f'Histogram of {y_col}', f'Box Plot of {y_col}', 
                                   f'Violin Plot of {y_col}', f'Density Plot of {y_col}'),
                    specs=[[{"colspan": 1}, {"colspan": 1}],
                           [{"colspan": 1}, {"colspan": 1}]]
                )
                
                # Histogram
                fig.add_trace(
                    go.Histogram(x=df[y_col], nbinsx=30, name="Histogram", opacity=0.7),
                    row=1, col=1
                )
                
                # Box plot
                fig.add_trace(
                    go.Box(y=df[y_col], name="Box Plot"),
                    row=1, col=2
                )
                
                # Violin plot
                fig.add_trace(
                    go.Violin(y=df[y_col], name="Violin Plot", box_visible=True),
                    row=2, col=1
                )
                
                # Density plot (using histogram with density)
                fig.add_trace(
                    go.Histogram(x=df[y_col], histnorm='probability density', 
                               nbinsx=50, name="Density", opacity=0.6),
                    row=2, col=2
                )
                
                fig.update_layout(height=800, title_text=f"Distribution Analysis of {y_col}")
                return fig
        
        elif chart_type == 'advanced_scatter':
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                fig = px.scatter(
                    df, x=x_col, y=y_col,
                    color=color_col if color_col and color_col in df.columns else None,
                    size=df[y_col] if df[y_col].min() >= 0 else None,
                    hover_data=[col for col in df.columns[:5]],
                    title=f"Advanced Scatter: {x_col} vs {y_col}",
                    trendline="ols" if df[x_col].dtype in ['int64', 'float64'] and df[y_col].dtype in ['int64', 'float64'] else None,
                    marginal_x="histogram",
                    marginal_y="box"
                )
                
                fig.update_traces(marker=dict(size=8, opacity=0.6))
                fig.update_layout(height=600)
                return fig
        
        elif chart_type == 'categorical_analysis':
            if x_col and x_col in df.columns:
                categorical_data = df[x_col].value_counts().head(15)
                
                # Create subplot with bar chart and pie chart
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=(f'Bar Chart of {x_col}', f'Pie Chart of {x_col}'),
                    specs=[[{"type": "bar"}, {"type": "pie"}]]
                )
                
                # Bar chart
                fig.add_trace(
                    go.Bar(x=categorical_data.index, y=categorical_data.values, 
                          name="Count", marker_color='lightblue'),
                    row=1, col=1
                )
                
                # Pie chart
                fig.add_trace(
                    go.Pie(labels=categorical_data.index, values=categorical_data.values,
                          name="Distribution"),
                    row=1, col=2
                )
                
                fig.update_layout(height=500, title_text=f"Categorical Analysis of {x_col}")
                return fig
        
        elif chart_type == 'time_series':
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                fig = go.Figure()
                
                # Line plot
                fig.add_trace(go.Scatter(
                    x=df[x_col], y=df[y_col],
                    mode='lines+markers',
                    name=f'{y_col} over {x_col}',
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
                
                # Add moving average if data is numeric
                if df[y_col].dtype in ['int64', 'float64'] and len(df) > 10:
                    rolling_mean = df[y_col].rolling(window=min(10, len(df)//3)).mean()
                    fig.add_trace(go.Scatter(
                        x=df[x_col], y=rolling_mean,
                        mode='lines',
                        name=f'{y_col} Moving Average',
                        line=dict(dash='dash', width=2)
                    ))
                
                fig.update_layout(
                    title=f"Time Series: {y_col} over {x_col}",
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    height=500
                )
                return fig
        
        elif chart_type == 'multi_variable':
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
            if len(numeric_cols) > 2:
                # Create parallel coordinates plot
                fig = go.Figure(data=
                    go.Parcoords(
                        line=dict(color=df[numeric_cols[0]], colorscale='Viridis'),
                        dimensions=list([
                            dict(range=[df[col].min(), df[col].max()],
                                label=col, values=df[col]) for col in numeric_cols
                        ])
                    )
                )
                
                fig.update_layout(
                    title="Multi-Variable Analysis (Parallel Coordinates)",
                    height=500
                )
                return fig
        
        elif chart_type == 'outlier_detection':
            if y_col and y_col in df.columns and df[y_col].dtype in ['int64', 'float64']:
                # Detect outliers
                outlier_info = detect_outliers(df, y_col)
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=(f'Box Plot with Outliers - {y_col}', f'Distribution with Outlier Bounds - {y_col}')
                )
                
                # Box plot
                fig.add_trace(
                    go.Box(y=df[y_col], name=y_col, boxpoints='outliers'),
                    row=1, col=1
                )
                
                # Histogram with outlier bounds
                fig.add_trace(
                    go.Histogram(x=df[y_col], name="Distribution", opacity=0.7),
                    row=1, col=2
                )
                
                # Add outlier bounds lines
                iqr_info = outlier_info['iqr_method']
                fig.add_vline(x=iqr_info['lower_bound'], line_dash="dash", line_color="red", 
                            annotation_text="Lower Bound", row=1, col=2)
                fig.add_vline(x=iqr_info['upper_bound'], line_dash="dash", line_color="red", 
                            annotation_text="Upper Bound", row=1, col=2)
                
                fig.update_layout(height=500, title_text=f"Outlier Analysis - {y_col}")
                return fig
        
        elif chart_type == 'pca_analysis':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 2:
                # Prepare data for PCA
                pca_data = df[numeric_cols].dropna()
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(pca_data)
                
                # Perform PCA
                pca = PCA(n_components=min(3, len(numeric_cols)))
                pca_result = pca.fit_transform(scaled_data)
                
                # Create 2D PCA plot
                fig = px.scatter(
                    x=pca_result[:, 0], 
                    y=pca_result[:, 1],
                    title=f"PCA Analysis - Explained Variance: {pca.explained_variance_ratio_[:2].sum():.2%}",
                    labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', 
                           'y': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'}
                )
                
                fig.update_layout(height=500)
                return fig
        
        return None
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def generate_comprehensive_insights_with_gemini(query, data_context, model):
    """Generate comprehensive insights using Gemini API with enhanced prompting"""
    try:
        # Create a detailed prompt with actual data context
        prompt = f"""
        You are an expert data scientist analyzing a dataset. Provide detailed, accurate insights based on the actual data provided.

        DATASET INFORMATION:
        - Shape: {data_context['basic_info']['shape'][0]:,} rows × {data_context['basic_info']['shape'][1]} columns
        - Total Data Points: {data_context['basic_info']['total_cells']:,}
        - Data Completeness: {data_context['data_quality']['completeness_score']:.1f}%
        - Duplicate Rows: {data_context['data_quality']['duplicate_rows']:,} ({data_context['data_quality']['duplicate_percentage']:.1f}%)

        COLUMNS AND DATA TYPES:
        {json.dumps(data_context['basic_info']['data_types'], indent=2)}

        NUMERIC COLUMNS STATISTICS:
        {json.dumps(data_context['statistical_summary'], indent=2)}

        CATEGORICAL COLUMNS ANALYSIS:
        {json.dumps(data_context['categorical_summary'], indent=2)}

        DATETIME COLUMNS ANALYSIS:
        {json.dumps(data_context.get('datetime_summary', {}), indent=2)}

        CORRELATION ANALYSIS:
        {json.dumps(data_context['correlation_analysis'], indent=2)}

        ADVANCED ANALYSIS:
        {json.dumps(data_context.get('advanced_analysis', {}), indent=2)}

        DATA QUALITY ISSUES:
        - Missing Values: {json.dumps(data_context['missing_values'], indent=2)}
        - High Null Columns (>50% missing): {data_context['data_quality']['high_null_columns']}
        - Constant Columns: {data_context['data_quality']['constant_columns']}

        USER QUERY: {query}

        Based on this ACTUAL data, provide comprehensive analysis. Use specific numbers from the data.

        Respond with a JSON object containing:
        {{
            "direct_answer": "Detailed answer with specific numbers and findings from the actual data",
            "key_insights": [
                "Insight 1 with specific numbers",
                "Insight 2 with actual data points", 
                "Insight 3 with statistical findings",
                "Insight 4 with correlation findings",
                "Insight 5 with data quality observations"
            ],
            "numerical_findings": [
                "Specific numerical finding 1",
                "Specific numerical finding 2",
                "Specific numerical finding 3"
            ],
            "statistical_summary": [
                "Statistical observation 1 with numbers",
                "Statistical observation 2 with numbers"
            ],
            "data_quality_assessment": [
                "Data quality finding 1",
                "Data quality finding 2"
            ],
            "visualization_recommendations": [
                {{
                    "chart_type": "chart_type_name",
                    "x_column": "column_name_or_null",
                    "y_column": "column_name_or_null", 
                    "color_column": "column_name_or_null",
                    "description": "Why this visualization is recommended",
                    "priority": "high/medium/low"
                }}
            ],
            "business_insights": [
                "Business insight 1",
                "Business insight 2"
            ],
            "follow_up_questions": [
                "Relevant follow-up question 1",
                "Relevant follow-up question 2"
            ],
            "primary_visualization": {{
                "chart_type": "recommended_primary_chart",
                "x_column": "column_name",
                "y_column": "column_name",
                "description": "Primary chart recommendation"
            }},
            "action_items": [
                "Recommended action 1",
                "Recommended action 2"
            ],
            "anomalies_detected": [
                "Anomaly or unusual pattern 1",
                "Anomaly or unusual pattern 2"
            ]
        }}

        IMPORTANT: Use actual numbers from the data provided. Be specific and quantitative in your analysis.
        """
        
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            try:
                parsed_response = json.loads(json_match.group())
                return parsed_response
            except json.JSONDecodeError:
                pass
        
        # Fallback with actual data
        numeric_cols = data_context['basic_info']['numeric_columns']
        categorical_cols = data_context['basic_info']['categorical_columns']
        datetime_cols = data_context['basic_info'].get('datetime_columns', [])
        
        return {
            "direct_answer": f"Analysis of your dataset with {data_context['basic_info']['shape'][0]:,} rows and {data_context['basic_info']['shape'][1]} columns. Data completeness is {data_context['data_quality']['completeness_score']:.1f}% with {len(numeric_cols)} numeric and {len(categorical_cols)} categorical variables.",
            "key_insights": [
                f"Dataset contains {data_context['basic_info']['shape'][0]:,} records across {data_context['basic_info']['shape'][1]} variables",
                f"Data completeness: {data_context['data_quality']['completeness_score']:.1f}% ({data_context['basic_info']['non_null_cells']:,} non-null values)",
                f"Found {data_context['data_quality']['duplicate_rows']} duplicate rows ({data_context['data_quality']['duplicate_percentage']:.1f}%)",
                f"Numeric variables: {len(numeric_cols)} - {', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''}",
                f"Categorical variables: {len(categorical_cols)} - {', '.join(categorical_cols[:3])}{'...' if len(categorical_cols) > 3 else ''}"
            ],
            "numerical_findings": [
                f"Total data points analyzed: {data_context['basic_info']['total_cells']:,}",
                f"Memory usage: {data_context['data_quality']['memory_usage_mb']:.2f} MB",
                f"Missing values: {sum(data_context['missing_values'].values()):,} total"
            ],
            "statistical_summary": [
                "Statistical analysis completed for all numeric variables",
                f"Correlation analysis performed on {len(numeric_cols)} numeric variables"
            ],
            "data_quality_assessment": [
                f"Overall data quality score: {data_context['data_quality']['completeness_score']:.1f}%",
                f"Identified {len(data_context['data_quality']['high_null_columns'])} columns with high missing values"
            ],
            "visualization_recommendations": [
                {
                    "chart_type": "overview_dashboard",
                    "x_column": None,
                    "y_column": None,
                    "color_column": None,
                    "description": "Comprehensive dashboard view of your data",
                    "priority": "high"
                },
                {
                    "chart_type": "correlation_heatmap" if len(numeric_cols) > 1 else "distribution_analysis",
                    "x_column": numeric_cols[0] if numeric_cols else None,
                    "y_column": numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0] if numeric_cols else None,
                    "color_column": None,
                    "description": "Correlation analysis between numeric variables" if len(numeric_cols) > 1 else "Distribution analysis of main variable",
                    "priority": "high"
                }
            ],
            "business_insights": [
                "Data is ready for further analysis and modeling",
                "Consider addressing missing values and duplicates for better analysis"
            ],
            "follow_up_questions": [
                "Show me the distribution of each numeric variable",
                "What are the strongest correlations in my data?",
                "Identify outliers in the numeric columns"
            ],
            "primary_visualization": {
                "chart_type": "overview_dashboard",
                "x_column": None,
                "y_column": None,
                "description": "Start with a comprehensive dashboard view"
            },
            "action_items": [
                "Clean data by handling missing values",
                "Remove or investigate duplicate records"
            ],
            "anomalies_detected": [
                f"High number of duplicates: {data_context['data_quality']['duplicate_percentage']:.1f}%" if data_context['data_quality']['duplicate_percentage'] > 5 else "No significant duplicates",
                f"Columns with >50% missing data: {len(data_context['data_quality']['high_null_columns'])}" if data_context['data_quality']['high_null_columns'] else "No severely incomplete columns"
            ]
        }
    
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return {
            "direct_answer": f"I encountered an error while analyzing your data: {str(e)}",
            "key_insights": ["Error in analysis - please try again"],
            "numerical_findings": ["Analysis could not be completed"],
            "statistical_summary": ["Please retry the analysis"],
            "data_quality_assessment": ["Analysis interrupted"],
            "visualization_recommendations": [],
            "business_insights": ["Please try a different question"],
            "follow_up_questions": ["Please rephrase your question"],
            "primary_visualization": {"chart_type": None, "x_column": None, "y_column": None, "description": "No visualization available"},
            "action_items": ["Retry the analysis"],
            "anomalies_detected": ["Analysis incomplete"]
        }

def parse_visualization_request(request, df, model):
    """Parse natural language visualization request using Gemini"""
    
    columns_info = {
        "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
        "datetime_columns": list(df.select_dtypes(include=['datetime64']).columns),
        "all_columns": list(df.columns)
    }
    
    prompt = f"""
    Given this visualization request: "{request}"
    
    And these available columns:
    Numeric: {columns_info['numeric_columns']}
    Categorical: {columns_info['categorical_columns']}
    Datetime: {columns_info['datetime_columns']}
    
    Determine the best visualization. Return a JSON object with:
    {{
        "chart_type": "one of: scatter, bar, line, histogram, box, pie, heatmap, distribution_analysis, categorical_analysis, time_series, correlation_heatmap",
        "x_column": "column name or null",
        "y_column": "column name or null",
        "color_column": "column name or null",
        "title": "descriptive title",
        "explanation": "why this visualization is appropriate"
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    # Fallback logic
    request_lower = request.lower()
    
    # Simple pattern matching
    if "correlation" in request_lower or "relationship" in request_lower:
        if len(columns_info['numeric_columns']) >= 2:
            return {
                "chart_type": "correlation_heatmap",
                "x_column": None,
                "y_column": None,
                "color_column": None,
                "title": "Correlation Analysis",
                "explanation": "Shows relationships between numeric variables"
            }
    
    if "distribution" in request_lower:
        col = None
        for num_col in columns_info['numeric_columns']:
            if num_col.lower() in request_lower:
                col = num_col
                break
        
        return {
            "chart_type": "distribution_analysis",
            "x_column": None,
            "y_column": col or columns_info['numeric_columns'][0] if columns_info['numeric_columns'] else None,
            "color_column": None,
            "title": f"Distribution of {col or 'data'}",
            "explanation": "Shows data distribution patterns"
        }
    
    # Default scatter plot for numeric data
    if len(columns_info['numeric_columns']) >= 2:
        return {
            "chart_type": "advanced_scatter",
            "x_column": columns_info['numeric_columns'][0],
            "y_column": columns_info['numeric_columns'][1],
            "color_column": columns_info['categorical_columns'][0] if columns_info['categorical_columns'] else None,
            "title": "Scatter Plot Analysis",
            "explanation": "Shows relationship between two numeric variables"
        }
    
    return None

def create_analysis_pdf(analysis_data, df_info):
    """Create a PDF report of the analysis"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    story.append(Paragraph("Data Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Date
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(Paragraph(analysis_data.get('direct_answer', 'No summary available'), styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Key Insights
    story.append(Paragraph("Key Insights", styles['Heading2']))
    for insight in analysis_data.get('key_insights', []):
        story.append(Paragraph(f"• {insight}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Numerical Findings
    story.append(Paragraph("Numerical Findings", styles['Heading2']))
    for finding in analysis_data.get('numerical_findings', []):
        story.append(Paragraph(f"• {finding}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Business Insights
    story.append(Paragraph("Business Insights", styles['Heading2']))
    for insight in analysis_data.get('business_insights', []):
        story.append(Paragraph(f"• {insight}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Action Items
    story.append(Paragraph("Recommended Actions", styles['Heading2']))
    for action in analysis_data.get('action_items', []):
        story.append(Paragraph(f"• {action}", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def get_visualization_suggestions(df, data_info):
    """Get intelligent visualization suggestions based on data characteristics"""
    suggestions = []
    
    numeric_cols = data_info['basic_info']['numeric_columns']
    categorical_cols = data_info['basic_info']['categorical_columns']
    datetime_cols = data_info['basic_info'].get('datetime_columns', [])
    
    # Distribution analysis suggestions
    if numeric_cols:
        suggestions.append({
            'prompt': f"Show me the distribution of {numeric_cols[0]}",
            'description': "Analyze the distribution pattern of your primary numeric variable",
            'chart_type': 'distribution_analysis',
            'params': {'y_col': numeric_cols[0]}
        })
    
    # Correlation suggestions
    if len(numeric_cols) > 1:
        suggestions.append({
            'prompt': f"Show correlation between {numeric_cols[0]} and {numeric_cols[1]}",
            'description': "Explore relationships between numeric variables",
            'chart_type': 'advanced_scatter',
            'params': {'x_col': numeric_cols[0], 'y_col': numeric_cols[1]}
        })
        
        suggestions.append({
            'prompt': "Create a correlation heatmap for all numeric variables",
            'description': "Visualize correlations between all numeric columns",
            'chart_type': 'correlation_heatmap',
            'params': {}
        })
    
    # Categorical analysis suggestions
    if categorical_cols:
        suggestions.append({
            'prompt': f"Show me the distribution of {categorical_cols[0]}",
            'description': "Analyze the frequency distribution of categorical data",
            'chart_type': 'categorical_analysis',
            'params': {'x_col': categorical_cols[0]}
        })
    
    # Time series suggestions
    if datetime_cols and numeric_cols:
        suggestions.append({
            'prompt': f"Show {numeric_cols[0]} over time using {datetime_cols[0]}",
            'description': "Analyze trends and patterns over time",
            'chart_type': 'time_series',
            'params': {'x_col': datetime_cols[0], 'y_col': numeric_cols[0]}
        })
    
    # Outlier detection suggestions
    if numeric_cols:
        suggestions.append({
            'prompt': f"Detect outliers in {numeric_cols[0]}",
            'description': "Identify unusual values and anomalies",
            'chart_type': 'outlier_detection',
            'params': {'y_col': numeric_cols[0]}
        })
    
    # Multi-variable analysis
    if len(numeric_cols) > 2:
        suggestions.append({
            'prompt': "Show multi-variable relationships",
            'description': "Explore patterns across multiple numeric variables",
            'chart_type': 'multi_variable',
            'params': {}
        })
        
        suggestions.append({
            'prompt': "Perform PCA analysis",
            'description': "Reduce dimensionality and find principal components",
            'chart_type': 'pca_analysis',
            'params': {}
        })
    
    return suggestions

def create_advanced_filters(df, data_info):
    """Create advanced filtering options based on data types"""
    filters = {}
    
    numeric_cols = data_info['basic_info']['numeric_columns']
    categorical_cols = data_info['basic_info']['categorical_columns']
    datetime_cols = data_info['basic_info'].get('datetime_columns', [])
    
    st.markdown("### 🎯 Advanced Filters")
    
    # Date filters
    if datetime_cols:
        st.markdown("**📅 Date Filters:**")
        for col in datetime_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                date_min = col_data.min()
                date_max = col_data.max()
                
                # Handle both date and datetime objects
                if hasattr(date_min, 'date'):
                    date_min = date_min.date()
                    date_max = date_max.date()
                
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    start_date = st.date_input(f"{col} - Start Date", value=date_min, min_value=date_min, max_value=date_max, key=f"start_{col}")
                with col2:
                    end_date = st.date_input(f"{col} - End Date", value=date_max, min_value=date_min, max_value=date_max, key=f"end_{col}")
                with col3:
                    if st.button("🔄 Reset", key=f"reset_date_{col}"):
                        start_date = date_min
                        end_date = date_max
                
                filters[col] = (pd.Timestamp(start_date), pd.Timestamp(end_date))
                
                # Quick date filters
                st.markdown("Quick filters:")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("Last 7 days", key=f"7d_{col}"):
                        filters[col] = (pd.Timestamp(date_max) - timedelta(days=7), pd.Timestamp(date_max))
                with col2:
                    if st.button("Last 30 days", key=f"30d_{col}"):
                        filters[col] = (pd.Timestamp(date_max) - timedelta(days=30), pd.Timestamp(date_max))
                with col3:
                    if st.button("Last 90 days", key=f"90d_{col}"):
                        filters[col] = (pd.Timestamp(date_max) - timedelta(days=90), pd.Timestamp(date_max))
                with col4:
                    if st.button("This year", key=f"year_{col}"):
                        if hasattr(date_max, 'year'):
                            current_year = date_max.year
                        else:
                            current_year = date_max.year
                        filters[col] = (pd.Timestamp(f"{current_year}-01-01"), pd.Timestamp(date_max))
                
                # Day of week filter
                st.markdown("**Day of Week Filter:**")
                days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                selected_days = st.multiselect(f"Select days for {col}", days_of_week, default=days_of_week, key=f"dow_{col}")
                filters[f"{col}_dow"] = [days_of_week.index(day) for day in selected_days]
    
    # Numeric range filters with percentile options
    if numeric_cols:
        st.markdown("**📊 Numeric Range Filters:**")
        for col in numeric_cols[:5]:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                col_min, col_max = float(col_data.min()), float(col_data.max())
                
                with st.expander(f"Filter {col}"):
                    # Percentile-based filtering
                    percentile_filter = st.checkbox(f"Use percentile filtering for {col}", key=f"perc_{col}")
                    
                    if percentile_filter:
                        percentile_range = st.slider(
                            f"Percentile range for {col}",
                            0, 100, (0, 100), 5,
                            key=f"perc_range_{col}"
                        )
                        lower_val = float(col_data.quantile(percentile_range[0] / 100))
                        upper_val = float(col_data.quantile(percentile_range[1] / 100))
                        filters[col] = (lower_val, upper_val)
                        st.info(f"Selected range: {lower_val:.2f} to {upper_val:.2f}")
                    else:
                        filters[col] = st.slider(
                            f"Range for {col}",
                            min_value=col_min,
                            max_value=col_max,
                            value=(col_min, col_max),
                            step=(col_max - col_min) / 100 if col_max > col_min else 0.01,
                            key=f"range_{col}"
                        )
                    
                    # Outlier filtering option
                    if st.checkbox(f"Exclude outliers from {col}", key=f"outlier_{col}"):
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        filters[col] = (float(Q1 - 1.5 * IQR), float(Q3 + 1.5 * IQR))
                        st.info(f"Outlier bounds: {filters[col][0]:.2f} to {filters[col][1]:.2f}")
    
    # Categorical filters with search
    if categorical_cols:
        st.markdown("**📋 Categorical Filters:**")
        for col in categorical_cols[:5]:
            unique_values = df[col].dropna().unique()
            if len(unique_values) <= 100:  # Only show if manageable number
                with st.expander(f"Filter {col} ({len(unique_values)} unique values)"):
                    # Search functionality for categories
                    search_term = st.text_input(f"Search in {col}", key=f"search_{col}")
                    
                    if search_term:
                        filtered_values = [val for val in unique_values if search_term.lower() in str(val).lower()]
                    else:
                        filtered_values = unique_values
                    
                    # Select all/none buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Select All", key=f"all_{col}"):
                            filters[col] = list(filtered_values)
                    with col2:
                        if st.button(f"Clear All", key=f"none_{col}"):
                            filters[col] = []
                    
                    filters[col] = st.multiselect(
                        f"Select values",
                        options=filtered_values,
                        default=list(filtered_values) if len(filtered_values) <= 10 else list(filtered_values[:10]),
                        key=f"multi_{col}"
                    )
    
    # Pattern-based filters
    st.markdown("**🔍 Pattern-based Filters:**")
    pattern_col = st.selectbox("Select column for pattern matching", ["None"] + list(df.columns))
    if pattern_col != "None":
        pattern = st.text_input("Enter pattern (regex supported)", key="pattern_input")
        if pattern:
            filters[f"{pattern_col}_pattern"] = pattern
    
    # Null value handling
    st.markdown("**❓ Missing Value Filters:**")
    null_handling = st.radio(
        "How to handle missing values:",
        ["Include all", "Exclude rows with any nulls", "Exclude rows with nulls in specific columns"],
        key="null_handling"
    )
    
    if null_handling == "Exclude rows with nulls in specific columns":
        null_cols = st.multiselect("Select columns to check for nulls", df.columns)
        filters['null_columns'] = null_cols
    elif null_handling == "Exclude rows with any nulls":
        filters['exclude_all_nulls'] = True
    
    return filters

def apply_advanced_filters(df, filters):
    """Apply advanced filters to dataframe"""
    filtered_df = df.copy()
    
    for col, filter_val in filters.items():
        if col in df.columns:
            if isinstance(filter_val, tuple) and len(filter_val) == 2:
                # Range filter for numeric or datetime
                if df[col].dtype in ['int64', 'float64']:
                    filtered_df = filtered_df[(filtered_df[col] >= filter_val[0]) & (filtered_df[col] <= filter_val[1])]
                elif df[col].dtype == 'datetime64[ns]' or pd.api.types.is_datetime64_any_dtype(df[col]):
                    filtered_df = filtered_df[(filtered_df[col] >= filter_val[0]) & (filtered_df[col] <= filter_val[1])]
            elif isinstance(filter_val, list):
                # Multi-select filter
                filtered_df = filtered_df[filtered_df[col].isin(filter_val)]
        elif col.endswith('_dow'):
            # Day of week filter
            date_col = col.replace('_dow', '')
            if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
                filtered_df = filtered_df[filtered_df[date_col].dt.dayofweek.isin(filter_val)]
        elif col.endswith('_pattern'):
            # Pattern matching
            pattern_col = col.replace('_pattern', '')
            if pattern_col in df.columns:
                filtered_df = filtered_df[filtered_df[pattern_col].astype(str).str.contains(filter_val, regex=True, na=False)]
        elif col == 'null_columns':
            # Null handling for specific columns
            filtered_df = filtered_df.dropna(subset=filter_val)
        elif col == 'exclude_all_nulls' and filter_val:
            # Exclude all nulls
            filtered_df = filtered_df.dropna()
    
    return filtered_df

def main():
    # Initialize everything
    initialize_session_state()
    model = initialize_gemini()
    
    # Header with custom styling
    st.markdown('<div class="main-header"><h1>🤖 Smart Data Analysis Chat</h1><p>Upload your CSV data and get comprehensive AI-powered insights!</p></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings & Upload")
        
        # Data upload
        uploaded_file = st.file_uploader(
            "📁 Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file to start analyzing your data"
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("🔄 Processing your data..."):
                    # Read CSV with error handling
                    df = pd.read_csv(uploaded_file)
                    
                    # Clean column names
                    df.columns = df.columns.str.strip()
                    
                    # Auto-detect and convert data types
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            # Try to convert to numeric
                            numeric_series = pd.to_numeric(df[col], errors='coerce')
                            if not numeric_series.isna().all():
                                df[col] = numeric_series
                            # Try to convert to datetime
                            elif df[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}').any():
                                try:
                                    df[col] = pd.to_datetime(df[col], errors='coerce')
                                except:
                                    pass
                    
                    # Store in session state
                    st.session_state.current_data = df
                    st.session_state.data_info = comprehensive_data_analysis(df)
                
                st.success(f"✅ Data loaded successfully!")
                
                # Enhanced data preview
                with st.expander("👀 Data Preview", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Data summary metrics
                with st.expander("📊 Quick Stats", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f'<div class="metric-container"><b>Rows:</b> {df.shape[0]:,}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-container"><b>Numeric Cols:</b> {len(st.session_state.data_info["basic_info"]["numeric_columns"])}</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="metric-container"><b>Columns:</b> {df.shape[1]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-container"><b>Categorical Cols:</b> {len(st.session_state.data_info["basic_info"]["categorical_columns"])}</div>', unsafe_allow_html=True)
                
                # Data quality indicators
                with st.expander("🔍 Data Quality", expanded=False):
                    completeness = st.session_state.data_info['data_quality']['completeness_score']
                    duplicates = st.session_state.data_info['data_quality']['duplicate_percentage']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Completeness", f"{completeness:.1f}%", 
                                delta=f"{completeness-80:.1f}%" if completeness > 80 else f"{completeness-80:.1f}%")
                    with col2:
                        st.metric("Duplicates", f"{duplicates:.1f}%",
                                delta=f"-{duplicates:.1f}%" if duplicates < 5 else f"{duplicates:.1f}%")
                    with col3:
                        memory_mb = st.session_state.data_info['data_quality']['memory_usage_mb']
                        st.metric("Memory Usage", f"{memory_mb:.1f} MB")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.info("💡 Please ensure your CSV file is properly formatted")
    
    # Main content area
    if st.session_state.current_data is not None:
        df = st.session_state.current_data
        data_info = st.session_state.data_info
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 AI Chat", "📊 Visualizations", "📈 Statistics", "🔍 Data Explorer", "📋 Summary Report"])
        
        with tab1:
            st.header("🤖 AI-Powered Data Analysis")
            
            # Enhanced quick action buttons
            st.subheader("🚀 Quick Analysis")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📊 Overall Summary", use_container_width=True):
                    st.session_state.chat_query = "Give me a comprehensive summary of this dataset including key statistics, patterns, and insights"
            
            with col2:
                if st.button("🔍 Find Patterns", use_container_width=True):
                    st.session_state.chat_query = "Identify interesting patterns, correlations, and anomalies in this data"
            
            with col3:
                if st.button("📈 Best Visualizations", use_container_width=True):
                    st.session_state.chat_query = "Recommend the best visualizations for this dataset and explain why"
            
            with col4:
                if st.button("⚠️ Data Quality Issues", use_container_width=True):
                    st.session_state.chat_query = "Analyze data quality issues, missing values, outliers, and suggest improvements"
            
            # Additional analysis options
            st.subheader("🎯 Advanced Analysis Options")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📊 Statistical Tests", use_container_width=True):
                    st.session_state.chat_query = "Perform relevant statistical tests on this data and interpret the results"
            
            with col2:
                if st.button("🎯 Predictive Insights", use_container_width=True):
                    st.session_state.chat_query = "What predictive insights can you derive from this data? What future trends might we expect?"
            
            with col3:
                if st.button("💼 Business Value", use_container_width=True):
                    st.session_state.chat_query = "What business value and actionable insights can be extracted from this data?"
            
            with col4:
                if st.button("🔄 Segmentation", use_container_width=True):
                    st.session_state.chat_query = "Identify potential segments or clusters in this data and describe their characteristics"
            
            # Chat interface
            st.subheader("💭 Ask Questions About Your Data")
            
            # Sample questions
            with st.expander("💡 Sample Questions"):
                sample_questions = [
                    "What are the main trends in my data?",
                    "Which variables are most strongly correlated?",
                    "Are there any outliers I should be concerned about?",
                    "What's the distribution of my key variables?",
                    "Can you identify any missing data patterns?",
                    "What business insights can you derive from this data?",
                    "Which visualization would best show my data relationships?",
                    "Are there any data quality issues I should address?",
                    "Can you perform a time series analysis on my data?",
                    "What statistical tests are appropriate for my data?",
                    "How can I segment my data for better insights?",
                    "What predictive models would work best with this data?"
                ]
                
                # Display questions in a grid
                cols = st.columns(2)
                for i, question in enumerate(sample_questions):
                    with cols[i % 2]:
                        if st.button(f"🔸 {question}", key=f"sample_{i}", use_container_width=True):
                            st.session_state.chat_query = question
            
            # Chat input
            user_query = st.text_area(
                "🎯 Ask me anything about your data:",
                value=getattr(st.session_state, 'chat_query', ''),
                placeholder="e.g., 'What are the strongest correlations in my data?' or 'Show me outliers in sales column'",
                height=100
            )
            
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
            with col2:
                if st.button("💾 Download Analysis", use_container_width=True, disabled=len(st.session_state.chat_history) == 0):
                    if st.session_state.chat_history:
                        latest_analysis = st.session_state.chat_history[-1]['insights']
                        pdf_buffer = create_analysis_pdf(latest_analysis, data_info)
                        st.download_button(
                            "📄 Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
            with col3:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
            
            # Process query
            if analyze_button and user_query:
                with st.spinner("🧠 AI is analyzing your data..."):
                    try:
                        # Generate insights using Gemini
                        insights = generate_comprehensive_insights_with_gemini(user_query, data_info, model)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'query': user_query,
                            'insights': insights,
                            'timestamp': datetime.now()
                        })
                        
                        # Store for visualization
                        st.session_state.current_insights = insights
                        
                    except Exception as e:
                        st.error(f"❌ Error analyzing data: {str(e)}")
            
            # Display chat history
            if st.session_state.chat_history:
                st.subheader("💬 Analysis Results")
                
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                    with st.container():
                        st.markdown(f"**🙋 Question:** {chat['query']}")
                        
                        insights = chat['insights']
                        
                        # Main answer
                        st.markdown(f'<div class="insight-box"><h4>🎯 Analysis Result</h4><p>{insights.get("direct_answer", "No analysis available")}</p></div>', unsafe_allow_html=True)
                        
                        # Create columns for different insight types
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Key insights
                            if insights.get('key_insights'):
                                st.markdown("**🔍 Key Insights:**")
                                for insight in insights['key_insights'][:5]:
                                    st.markdown(f"• {insight}")
                            
                            # Numerical findings
                            if insights.get('numerical_findings'):
                                st.markdown("**📊 Numerical Findings:**")
                                for finding in insights['numerical_findings']:
                                    st.markdown(f"• {finding}")
                        
                        with col2:
                            # Business insights
                            if insights.get('business_insights'):
                                st.markdown("**💼 Business Insights:**")
                                for insight in insights['business_insights']:
                                    st.markdown(f"• {insight}")
                            
                            # Action items
                            if insights.get('action_items'):
                                st.markdown("**✅ Action Items:**")
                                for action in insights['action_items']:
                                    st.markdown(f"• {action}")
                        
                        # Anomalies detected
                        if insights.get('anomalies_detected'):
                            st.warning("**⚠️ Anomalies Detected:**")
                            for anomaly in insights['anomalies_detected']:
                                st.markdown(f"• {anomaly}")
                        
                        # Show primary visualization if recommended
                        if insights.get('primary_visualization') and insights['primary_visualization'].get('chart_type'):
                            viz_rec = insights['primary_visualization']
                            if viz_rec['chart_type']:
                                st.markdown("**📈 Recommended Visualization:**")
                                fig = create_comprehensive_visualizations(
                                    df, 
                                    viz_rec['chart_type'],
                                    viz_rec.get('x_column'),
                                    viz_rec.get('y_column'),
                                    f"Analysis: {chat['query'][:50]}..."
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True, key=f"chat_viz_{i}")
                        
                        # Follow-up questions
                        if insights.get('follow_up_questions'):
                            st.markdown("**🤔 Suggested Follow-up Questions:**")
                            cols = st.columns(len(insights['follow_up_questions'][:3]))
                            for j, follow_up in enumerate(insights['follow_up_questions'][:3]):
                                with cols[j]:
                                    if st.button(f"🔸 {follow_up[:30]}{'...' if len(follow_up) > 30 else ''}", 
                                               key=f"followup_{i}_{j}"):
                                        st.session_state.chat_query = follow_up
                                        st.rerun()
                        
                        st.markdown("---")
        
        with tab2:
            st.header("📊 Interactive Visualizations")
            
            # NEW: Natural Language Visualization Input
            st.subheader("🎨 Create Visualizations with Natural Language")
            viz_request = st.text_input(
                "Describe the chart you want to create:",
                placeholder="e.g., 'Show me a scatter plot of price vs quantity' or 'Create a bar chart of sales by category'"
            )
            
            if st.button("🪄 Create Visualization", type="primary") and viz_request:
                with st.spinner("Creating your visualization..."):
                    viz_params = parse_visualization_request(viz_request, df, model)
                    
                    if viz_params:
                        fig = create_comprehensive_visualizations(
                            df,
                            viz_params['chart_type'],
                            viz_params.get('x_column'),
                            viz_params.get('y_column'),
                            viz_params.get('title', viz_request),
                            viz_params.get('color_column')
                        )
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key=get_unique_key())
                            
                            # Explain the visualization
                            with st.expander("📝 Visualization Explanation"):
                                st.markdown(viz_params.get('explanation', 'Visualization created based on your request.'))
                        else:
                            st.warning("Could not create the requested visualization. Please try a different description.")
                    else:
                        st.error("Could not understand the visualization request. Please be more specific.")
            
            st.markdown("---")
            
            # Visualization suggestions
            st.subheader("💡 Visualization Suggestions")
            suggestions = get_visualization_suggestions(df, data_info)
            
            if suggestions:
                st.markdown("**Click on a suggestion to create the visualization:**")
                
                # Display suggestions in a grid
                cols = st.columns(3)
                for i, suggestion in enumerate(suggestions):
                    with cols[i % 3]:
                        if st.button(
                            f"📊 {suggestion['prompt'][:40]}...",
                            help=suggestion['description'],
                            key=f"viz_sug_{i}",
                            use_container_width=True
                        ):
                            # Create the suggested visualization
                            with st.spinner("Creating visualization..."):
                                fig = create_comprehensive_visualizations(
                                    df, 
                                    suggestion['chart_type'],
                                    suggestion['params'].get('x_col'),
                                    suggestion['params'].get('y_col'),
                                    suggestion['prompt']
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True, key=get_unique_key())
                                    
                                    # Store in visualization history
                                    st.session_state.visualization_history.append({
                                        'prompt': suggestion['prompt'],
                                        'chart_type': suggestion['chart_type'],
                                        'timestamp': datetime.now()
                                    })
            
            st.markdown("---")
            
            # Custom visualization builder
            st.subheader("🛠️ Custom Visualization Builder")
            
            # Visualization controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                viz_type = st.selectbox(
                    "📈 Chart Type",
                    ["overview_dashboard", "correlation_heatmap", "distribution_analysis", 
                     "advanced_scatter", "categorical_analysis", "time_series", 
                     "multi_variable", "outlier_detection", "pca_analysis"],
                    format_func=lambda x: {
                        "overview_dashboard": "📊 Overview Dashboard",
                        "correlation_heatmap": "🔥 Correlation Heatmap", 
                        "distribution_analysis": "📈 Distribution Analysis",
                        "advanced_scatter": "🎯 Advanced Scatter Plot",
                        "categorical_analysis": "📋 Categorical Analysis",
                        "time_series": "📅 Time Series",
                        "multi_variable": "🌐 Multi-Variable Analysis",
                        "outlier_detection": "⚠️ Outlier Detection",
                        "pca_analysis": "🔬 PCA Analysis"
                    }.get(x, x)
                )
            
            with col2:
                numeric_cols = data_info['basic_info']['numeric_columns']
                categorical_cols = data_info['basic_info']['categorical_columns']
                datetime_cols = data_info['basic_info'].get('datetime_columns', [])
                all_cols = df.columns.tolist()
                
                x_col = st.selectbox("🔤 X-Axis Column", [None] + all_cols, 
                                   index=1 if len(all_cols) > 0 else 0)
            
            with col3:
                y_col = st.selectbox("📊 Y-Axis Column", [None] + numeric_cols,
                                   index=1 if len(numeric_cols) > 0 else 0)
            
            # Additional options
            col1, col2 = st.columns(2)
            with col1:
                color_col = st.selectbox("🎨 Color Column (Optional)", [None] + categorical_cols + numeric_cols[:3])
            with col2:
                chart_title = st.text_input("📝 Chart Title (Optional)", "")
            
            # Generate visualization
            if st.button("🎨 Generate Visualization", type="primary"):
                with st.spinner("Creating visualization..."):
                    fig = create_comprehensive_visualizations(
                        df, viz_type, x_col, y_col, 
                        chart_title or f"{viz_type.replace('_', ' ').title()}", 
                        color_col
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=get_unique_key())
                        
                        # Provide insights about the visualization
                        with st.expander("🔍 Visualization Insights"):
                            viz_query = f"Analyze this {viz_type.replace('_', ' ')} visualization showing {x_col} vs {y_col if y_col else 'data distribution'}. What patterns, trends, or insights can you identify?"
                            viz_insights = generate_comprehensive_insights_with_gemini(viz_query, data_info, model)
                            st.markdown(viz_insights.get('direct_answer', 'Analysis not available'))
            
            # Visualization history
            if st.session_state.visualization_history:
                st.markdown("---")
                st.subheader("📜 Recent Visualizations")
                for viz in st.session_state.visualization_history[-3:]:
                    st.markdown(f"• **{viz['prompt']}** - {viz['chart_type']} ({viz['timestamp'].strftime('%H:%M:%S')})")
            
            # Quick visualization grid
            st.markdown("---")
            st.subheader("🚀 Quick Visualizations")
            
            if numeric_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution of first numeric column
                    if len(numeric_cols) > 0:
                        fig_dist = create_comprehensive_visualizations(df, 'distribution_analysis', 
                                                                     None, numeric_cols[0], 
                                                                     f"Distribution of {numeric_cols[0]}")
                        if fig_dist:
                            st.plotly_chart(fig_dist, use_container_width=True, key=get_unique_key())
                
                with col2:
                    # Correlation heatmap if multiple numeric columns
                    if len(numeric_cols) > 1:
                        fig_corr = create_comprehensive_visualizations(df, 'correlation_heatmap')
                        if fig_corr:
                            st.plotly_chart(fig_corr, use_container_width=True, key=get_unique_key())
            
            # Categorical analysis
            if categorical_cols:
                st.subheader("📋 Categorical Data Analysis")
                selected_cat = st.selectbox("Select categorical column:", categorical_cols)
                if selected_cat:
                    fig_cat = create_comprehensive_visualizations(df, 'categorical_analysis', 
                                                                selected_cat, None, 
                                                                f"Analysis of {selected_cat}")
                    if fig_cat:
                        st.plotly_chart(fig_cat, use_container_width=True, key=get_unique_key())
        
        with tab3:
            st.header("📈 Statistical Analysis")
            
            # Enhanced statistical analysis sections
            tabs_stats = st.tabs(["📊 Descriptive Stats", "🔗 Correlations", "📐 Advanced Stats", "⚠️ Outliers"])
            
            with tabs_stats[0]:
                # Summary statistics
                if numeric_cols:
                    st.subheader("📊 Descriptive Statistics")
                    
                    # Enhanced statistics table
                    stats_df = df[numeric_cols].describe().round(3)
                    
                    # Add additional statistics
                    additional_stats = pd.DataFrame({
                        col: {
                            'skewness': df[col].skew(),
                            'kurtosis': df[col].kurtosis(),
                            'variance': df[col].var(),
                            'std_error': df[col].sem(),
                            'range': df[col].max() - df[col].min(),
                            'iqr': df[col].quantile(0.75) - df[col].quantile(0.25),
                            'cv': df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                        } for col in numeric_cols
                    }).round(3)
                    
                    # Combine statistics
                    full_stats = pd.concat([stats_df, additional_stats])
                    st.dataframe(full_stats, use_container_width=True)
                    
                    # Download statistics
                    csv = full_stats.to_csv()
                    st.download_button(
                        "📥 Download Statistics",
                        csv,
                        "statistics.csv",
                        "text/csv"
                    )
                    
                    # Statistical insights
                    with st.expander("🧠 Statistical Insights"):
                        stats_query = "Provide detailed statistical insights about the numeric variables including distribution characteristics, skewness, outliers, and what these statistics tell us about the data"
                        stats_insights = generate_comprehensive_insights_with_gemini(stats_query, data_info, model)
                        st.markdown(stats_insights.get('direct_answer', 'Statistical analysis not available'))
            
            with tabs_stats[1]:
                # Correlation analysis
                if len(numeric_cols) > 1:
                    st.subheader("🔗 Correlation Analysis")
                    
                    corr_matrix = df[numeric_cols].corr()
                    
                    # Display correlation matrix
                    fig_corr = px.imshow(corr_matrix.round(3), 
                                       text_auto=True, 
                                       aspect="auto",
                                       title="Correlation Matrix",
                                       color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig_corr, use_container_width=True, key=get_unique_key())
                    
                    # Strong correlations
                    strong_corr = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.5:
                                strong_corr.append({
                                    'Variable 1': corr_matrix.columns[i],
                                    'Variable 2': corr_matrix.columns[j],
                                    'Correlation': corr_val,
                                    'Strength': 'Very Strong' if abs(corr_val) > 0.8 else 'Strong' if abs(corr_val) > 0.7 else 'Moderate',
                                    'Direction': 'Positive' if corr_val > 0 else 'Negative'
                                })
                    
                    if strong_corr:
                        st.subheader("💪 Significant Correlations")
                        corr_df = pd.DataFrame(strong_corr).sort_values('Correlation', key=abs, ascending=False)
                        st.dataframe(corr_df, use_container_width=True)
                        
                        # Visualize strongest correlation
                        if len(corr_df) > 0:
                            strongest = corr_df.iloc[0]
                            fig_scatter = create_comprehensive_visualizations(
                                df, 'advanced_scatter',
                                strongest['Variable 1'], strongest['Variable 2'],
                                f"Strongest Correlation: {strongest['Variable 1']} vs {strongest['Variable 2']}"
                            )
                            if fig_scatter:
                                st.plotly_chart(fig_scatter, use_container_width=True, key=get_unique_key())
            
            with tabs_stats[2]:
                # Advanced statistical analysis
                st.subheader("📐 Advanced Statistical Analysis")
                
                if 'advanced_analysis' in data_info:
                    advanced_stats = data_info['advanced_analysis']
                    
                    # Normality tests
                    if 'normality_tests' in advanced_stats:
                        st.markdown("**🔔 Normality Tests (Shapiro-Wilk / D'Agostino-Pearson):**")
                        norm_df = pd.DataFrame(advanced_stats['normality_tests']).T
                        st.dataframe(norm_df, use_container_width=True)
                    
                    # Trend analysis
                    if 'trend_analysis' in advanced_stats:
                        st.markdown("**📈 Trend Analysis:**")
                        trend_df = pd.DataFrame(advanced_stats['trend_analysis']).T
                        st.dataframe(trend_df, use_container_width=True)
                    
                    # Variance analysis
                    if 'variance_analysis' in advanced_stats:
                        st.markdown("**📊 Variance Analysis:**")
                        if advanced_stats['variance_analysis']['high_variance']:
                            st.warning("High variance columns (CV > 1):")
                            for col, cv in advanced_stats['variance_analysis']['high_variance']:
                                st.markdown(f"• {col}: CV = {cv:.3f}")
                        if advanced_stats['variance_analysis']['low_variance']:
                            st.info("Low variance columns (CV < 0.1):")
                            for col, cv in advanced_stats['variance_analysis']['low_variance']:
                                st.markdown(f"• {col}: CV = {cv:.3f}")
            
            with tabs_stats[3]:
                # Outlier analysis
                st.subheader("⚠️ Outlier Detection")
                
                outlier_col = st.selectbox("Select column for outlier analysis:", numeric_cols)
                
                if outlier_col:
                    outliers = detect_outliers(df, outlier_col)
                    
                    # Display outlier statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("IQR Method", f"{outliers['iqr_method']['count']} outliers", 
                                f"{outliers['iqr_method']['percentage']:.1f}%")
                    with col2:
                        st.metric("Z-Score Method", f"{outliers['z_score_method']['count']} outliers",
                                f"{outliers['z_score_method']['percentage']:.1f}%")
                    with col3:
                        st.metric("Modified Z-Score", f"{outliers['modified_z_score']['count']} outliers",
                                f"{outliers['modified_z_score']['percentage']:.1f}%")
                    
                    # Visualize outliers
                    fig_outliers = create_comprehensive_visualizations(
                        df, 'outlier_detection',
                        None, outlier_col,
                        f"Outlier Analysis - {outlier_col}"
                    )
                    if fig_outliers:
                        st.plotly_chart(fig_outliers, use_container_width=True, key=get_unique_key())
                    
                    # Show outlier values
                    with st.expander("View Outlier Values"):
                        st.markdown("**IQR Method Outliers (first 10):**")
                        st.write(outliers['iqr_method']['outliers'][:10])
                        st.markdown(f"Bounds: [{outliers['iqr_method']['lower_bound']:.3f}, {outliers['iqr_method']['upper_bound']:.3f}]")
            
            # Missing values analysis
            if data_info['missing_values']:
                st.markdown("---")
                st.subheader("❓ Missing Values Analysis")
                
                missing_data = []
                for col, missing_count in data_info['missing_values'].items():
                    if missing_count > 0:
                        missing_data.append({
                            'Column': col,
                            'Missing Count': missing_count,
                            'Missing %': round(missing_count / len(df) * 100, 2),
                            'Data Type': str(df[col].dtype)
                        })
                
                if missing_data:
                    missing_df = pd.DataFrame(missing_data).sort_values('Missing %', ascending=False)
                    st.dataframe(missing_df, use_container_width=True)
                    
                    # Missing values heatmap
                    if len(missing_data) > 1:
                        missing_matrix = df.isnull()
                        fig_missing = px.imshow(missing_matrix.T, 
                                              title="Missing Values Pattern",
                                              labels=dict(x="Row Index", y="Columns"),
                                              color_continuous_scale=['white', 'red'])
                        st.plotly_chart(fig_missing, use_container_width=True, key=get_unique_key())
        
        with tab4:
            st.header("🔍 Data Explorer")
            
            # Advanced filtering section
            filters = create_advanced_filters(df, data_info)
            
            # Apply filters
            filtered_df = apply_advanced_filters(df, filters)
            
            # Show filtered results
            st.markdown("---")
            st.subheader(f"📋 Filtered Data ({len(filtered_df)} rows)")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{len(filtered_df):,}", f"{len(filtered_df) - len(df):,}")
            with col2:
                st.metric("% of Original", f"{len(filtered_df)/len(df)*100:.1f}%")
            with col3:
                st.metric("Filtered Out", f"{len(df) - len(filtered_df):,}")
            with col4:
                if st.button("🔄 Reset Filters"):
                    st.rerun()
            
            # Display filtered data with pagination
            rows_per_page = st.slider("Rows per page", 10, 100, 25)
            page_num = st.number_input("Page", min_value=1, max_value=max(1, len(filtered_df) // rows_per_page + 1), value=1)
            
            start_idx = (page_num - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(filtered_df))
            
            st.dataframe(filtered_df.iloc[start_idx:end_idx], use_container_width=True)
            
            # Download filtered data
            if len(filtered_df) > 0:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Filtered Data",
                    data=csv,
                    file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Quick analysis of filtered data
            if st.button("🔍 Analyze Filtered Data") and len(filtered_df) > 0:
                with st.spinner("Analyzing filtered data..."):
                    filtered_info = comprehensive_data_analysis(filtered_df)
                    filter_query = f"Analyze this filtered dataset with {len(filtered_df)} rows. Compare it with the original dataset and highlight key differences, patterns, and insights."
                    filter_insights = generate_comprehensive_insights_with_gemini(filter_query, filtered_info, model)
                    
                    st.markdown("**🎯 Filtered Data Insights:**")
                    st.markdown(filter_insights.get('direct_answer', 'Analysis not available'))
                    
                    # Show key metrics comparison
                    st.markdown("**📊 Key Metrics Comparison:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Original Data:**")
                        st.markdown(f"• Rows: {len(df):,}")
                        st.markdown(f"• Completeness: {data_info['data_quality']['completeness_score']:.1f}%")
                        if numeric_cols:
                            st.markdown(f"• Mean of {numeric_cols[0]}: {df[numeric_cols[0]].mean():.2f}")
                    
                    with col2:
                        st.markdown("**Filtered Data:**")
                        st.markdown(f"• Rows: {len(filtered_df):,}")
                        st.markdown(f"• Completeness: {filtered_info['data_quality']['completeness_score']:.1f}%")
                        if numeric_cols and numeric_cols[0] in filtered_df.columns:
                            st.markdown(f"• Mean of {numeric_cols[0]}: {filtered_df[numeric_cols[0]].mean():.2f}")
            
            # Data transformation options
            st.markdown("---")
            st.subheader("🔄 Data Transformations")
            
            transform_col = st.selectbox("Select column to transform:", df.columns)
            transform_type = st.selectbox(
                "Select transformation:",
                ["None", "Log Transform", "Square Root", "Standardize", "Normalize", "Bin Numeric"]
            )
            
            if transform_type != "None" and st.button("Apply Transformation"):
                try:
                    transformed_df = filtered_df.copy()
                    
                    if transform_type == "Log Transform" and df[transform_col].dtype in ['int64', 'float64']:
                        transformed_df[f"{transform_col}_log"] = np.log1p(transformed_df[transform_col])
                        st.success(f"Created {transform_col}_log")
                    elif transform_type == "Square Root" and df[transform_col].dtype in ['int64', 'float64']:
                        transformed_df[f"{transform_col}_sqrt"] = np.sqrt(np.abs(transformed_df[transform_col]))
                        st.success(f"Created {transform_col}_sqrt")
                    elif transform_type == "Standardize" and df[transform_col].dtype in ['int64', 'float64']:
                        scaler = StandardScaler()
                        transformed_df[f"{transform_col}_std"] = scaler.fit_transform(transformed_df[[transform_col]])
                        st.success(f"Created {transform_col}_std")
                    elif transform_type == "Normalize" and df[transform_col].dtype in ['int64', 'float64']:
                        min_val = transformed_df[transform_col].min()
                        max_val = transformed_df[transform_col].max()
                        if max_val > min_val:
                            transformed_df[f"{transform_col}_norm"] = (transformed_df[transform_col] - min_val) / (max_val - min_val)
                            st.success(f"Created {transform_col}_norm")
                        else:
                            st.warning("Cannot normalize: all values are the same")
                    elif transform_type == "Bin Numeric" and df[transform_col].dtype in ['int64', 'float64']:
                        n_bins = st.number_input("Number of bins:", min_value=2, max_value=20, value=5)
                        transformed_df[f"{transform_col}_binned"] = pd.cut(transformed_df[transform_col], bins=n_bins)
                        st.success(f"Created {transform_col}_binned with {n_bins} bins")
                    
                    st.dataframe(transformed_df.head(20), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error applying transformation: {str(e)}")
        
        with tab5:
            st.header("📋 Comprehensive Data Report")
            
            # Report generation options
            col1, col2, col3 = st.columns(3)
            with col1:
                report_type = st.selectbox(
                    "Report Type:",
                    ["Executive Summary", "Technical Report", "Full Analysis", "Custom Report"]
                )
            with col2:
                include_viz = st.checkbox("Include Visualizations", value=True)
            with col3:
                include_stats = st.checkbox("Include Full Statistics", value=True)
            
            # Generate comprehensive report
            if st.button("📊 Generate Report", type="primary"):
                with st.spinner("🔄 Generating comprehensive report..."):
                    
                    # Report sections
                    st.markdown("## 📋 Data Analysis Report")
                    st.markdown(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    st.markdown("---")
                    
                    # Executive Summary
                    st.markdown("### 🎯 Executive Summary")
                    summary_query = "Provide an executive summary of this dataset suitable for business stakeholders, highlighting key findings, data quality, and business implications"
                    summary_insights = generate_comprehensive_insights_with_gemini(summary_query, data_info, model)
                    st.markdown(summary_insights.get('direct_answer', 'Summary not available'))
                    
                    # Key findings
                    if summary_insights.get('key_insights'):
                        st.markdown("**Key Findings:**")
                        for finding in summary_insights['key_insights']:
                            st.markdown(f"• {finding}")
                    
                    # Data Overview
                    st.markdown("### 📊 Data Overview")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Records", f"{df.shape[0]:,}")
                    with col2:
                        st.metric("Total Columns", f"{df.shape[1]}")
                    with col3:
                        st.metric("Numeric Columns", f"{len(numeric_cols)}")
                    with col4:
                        st.metric("Categorical Columns", f"{len(categorical_cols)}")
                    
                    # Data Quality Assessment
                    st.markdown("### 🔍 Data Quality Assessment")
                    quality_metrics = data_info['data_quality']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Completeness", f"{quality_metrics['completeness_score']:.1f}%")
                    with col2:
                        st.metric("Duplicate Rows", f"{quality_metrics['duplicate_rows']:,}")
                    with col3:
                        st.metric("Memory Usage", f"{quality_metrics['memory_usage_mb']:.1f} MB")
                    
                    # Quality issues
                    if quality_metrics['high_null_columns'] or quality_metrics['constant_columns']:
                        st.warning("**Data Quality Issues:**")
                        if quality_metrics['high_null_columns']:
                            st.markdown(f"• Columns with >50% missing: {', '.join(quality_metrics['high_null_columns'])}")
                        if quality_metrics['constant_columns']:
                            st.markdown(f"• Constant columns: {', '.join(quality_metrics['constant_columns'])}")
                    
                    # Key Statistics
                    if include_stats and numeric_cols:
                        st.markdown("### 📈 Statistical Summary")
                        summary_stats = df[numeric_cols].describe().round(2)
                        st.dataframe(summary_stats, use_container_width=True)
                        
                        # Advanced statistics
                        st.markdown("**Advanced Statistics:**")
                        if 'advanced_analysis' in data_info:
                            advanced = data_info['advanced_analysis']
                            if 'normality_tests' in advanced:
                                normal_cols = [col for col, test in advanced['normality_tests'].items() if test['is_normal']]
                                non_normal_cols = [col for col, test in advanced['normality_tests'].items() if not test['is_normal']]
                                if normal_cols:
                                    st.success(f"Normally distributed columns: {', '.join(normal_cols[:5])}")
                                if non_normal_cols:
                                    st.warning(f"Non-normally distributed columns: {', '.join(non_normal_cols[:5])}")
                    
                    # Visualizations
                    if include_viz:
                        st.markdown("### 📊 Key Visualizations")
                        
                        # Overview dashboard
                        fig_overview = create_comprehensive_visualizations(df, 'overview_dashboard')
                        if fig_overview:
                            st.plotly_chart(fig_overview, use_container_width=True, key=get_unique_key())
                        
                        # Additional visualizations based on data type
                        if len(numeric_cols) > 2:
                            fig_pca = create_comprehensive_visualizations(df, 'pca_analysis')
                            if fig_pca:
                                st.markdown("**Principal Component Analysis:**")
                                st.plotly_chart(fig_pca, use_container_width=True, key=get_unique_key())
                    
                    # Business Insights
                    st.markdown("### 💼 Business Insights & Recommendations")
                    business_query = "Provide detailed business insights and actionable recommendations based on this data analysis. Focus on practical applications and value creation."
                    business_insights = generate_comprehensive_insights_with_gemini(business_query, data_info, model)
                    
                    if business_insights.get('business_insights'):
                        st.markdown("**Business Insights:**")
                        for insight in business_insights['business_insights']:
                            st.info(f"💡 {insight}")
                    
                    if business_insights.get('action_items'):
                        st.markdown("**Recommended Actions:**")
                        for i, action in enumerate(business_insights['action_items'], 1):
                            st.markdown(f"{i}. {action}")
                    
                    # Technical Details
                    with st.expander("🔧 Technical Details"):
                        st.markdown("**Column Information:**")
                        col_info = pd.DataFrame({
                            'Column': df.columns,
                            'Data Type': [str(dtype) for dtype in df.dtypes],
                            'Non-Null Count': [df[col].count() for col in df.columns],
                            'Null Count': [df[col].isnull().sum() for col in df.columns],
                            'Unique Values': [df[col].nunique() for col in df.columns]
                        })
                        st.dataframe(col_info, use_container_width=True)
                        
                        # Correlation details
                        if len(numeric_cols) > 1 and 'correlation_analysis' in data_info:
                            st.markdown("**Correlation Analysis:**")
                            corr_data = data_info['correlation_analysis']
                            if corr_data.get('strong_positive'):
                                st.success(f"Strong positive correlations: {len(corr_data['strong_positive'])}")
                            if corr_data.get('strong_negative'):
                                st.warning(f"Strong negative correlations: {len(corr_data['strong_negative'])}")
                    
                    # Save report
                    st.session_state.analysis_reports.append({
                        'timestamp': datetime.now(),
                        'type': report_type,
                        'summary': summary_insights,
                        'business': business_insights
                    })
            
            # Export options
            st.markdown("### 💾 Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Export Summary Stats"):
                    if numeric_cols:
                        stats_csv = df[numeric_cols].describe().to_csv()
                        st.download_button(
                            "Download Stats CSV",
                            stats_csv,
                            f"summary_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        )
            
            with col2:
                if st.button("🔗 Export Correlations"):
                    if len(numeric_cols) > 1:
                        corr_csv = df[numeric_cols].corr().to_csv()
                        st.download_button(
                            "Download Correlations CSV",
                            corr_csv,
                            f"correlations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        )
            
            with col3:
                if st.button("📊 Export Full Dataset"):
                    full_csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Full Dataset",
                        full_csv,
                        f"full_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    )
            
            # Report history
            if st.session_state.analysis_reports:
                st.markdown("### 📜 Report History")
                for report in st.session_state.analysis_reports[-3:]:
                    st.markdown(f"• **{report['type']}** - {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Welcome to Smart Data Analysis Chat!
        
        This powerful tool combines AI-driven insights with interactive visualizations to help you understand your data better.
        
        ### ✨ Enhanced Features:
        - 🤖 **AI-Powered Analysis**: Get intelligent insights using Google's Gemini AI
        - 📊 **Smart Visualization Suggestions**: AI-recommended charts based on your data
        - 🎨 **Natural Language Visualization**: Create charts by simply describing what you want
        - 📈 **Advanced Statistical Analysis**: Normality tests, outlier detection, trend analysis
        - 🔍 **Advanced Data Filtering**: Date ranges, day-of-week filters, percentile-based, pattern matching
        - 💾 **Downloadable Analysis Reports**: Export insights as PDF or CSV
        - 🎯 **Business Intelligence**: Actionable insights and recommendations
        - 📐 **PCA & Advanced Analytics**: Dimensionality reduction and complex analysis
        
        ### 🏁 Getting Started:
        1. **Upload your CSV file** using the sidebar
        2. **Explore AI suggestions** for visualizations and analysis
        3. **Ask questions** using natural language in the AI Chat tab
        4. **Create visualizations** by describing what you want to see
        5. **Apply advanced filters** including date and day-wise filtering
        6. **Download reports** and insights for sharing
        
        ### 💡 New Features in This Version:
        - **Natural Language Visualization Input**: Describe the chart you want in plain English
        - **Enhanced Date Filtering**: Filter by date ranges and specific days of the week
        - **Smart Chart Generation**: AI understands your visualization requests
        - **Unique Chart IDs**: Fixed duplicate chart element errors
        - **Improved Error Handling**: Better JSON serialization and module handling
        
        **👈 Start by uploading a CSV file in the sidebar!**
        """)
        
        # Sample data info
        with st.expander("📁 Supported File Formats & Requirements"):
            st.markdown("""
            **Supported Formats:**
            - CSV files (.csv)
            
            **Requirements:**
            - File size: Up to 200MB
            - Encoding: UTF-8 recommended
            - Headers: First row should contain column names
            - Data types: Automatic detection of numeric, categorical, and date columns
            
            **Tips for Best Results:**
            - Clean column names (avoid special characters)
            - Consistent date formats (YYYY-MM-DD or MM/DD/YYYY)
            - Numeric data should be properly formatted
            - Missing values can be represented as empty cells or 'NULL'
            
            **Advanced Features Available:**
            - Natural language chart creation
            - Automatic date detection and time series analysis
            - Day-of-week filtering for temporal analysis
            - Statistical testing and normality checks
            - Multi-method outlier detection
            - Pattern-based filtering with regex support
            - PCA for high-dimensional data
            """)

if __name__ == "__main__":
    main()