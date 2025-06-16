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
from datetime import datetime
import warnings
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

def comprehensive_data_analysis(df):
    """Comprehensive data structure analysis with detailed statistics"""
    
    # Basic info
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(include=['object', 'category', 'bool']).columns)
    datetime_cols = list(df.select_dtypes(include=['datetime64']).columns)
    
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
                'top_5_values': value_counts.head().to_dict(),
                'distribution_evenness': float(value_counts.std() / value_counts.mean()) if value_counts.mean() > 0 else 0
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
    
    return {
        'basic_info': {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.astype(str).to_dict(),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': datetime_cols,
            'total_cells': df.shape[0] * df.shape[1],
            'non_null_cells': int(df.count().sum())
        },
        'statistical_summary': stats_summary,
        'categorical_summary': categorical_summary,
        'correlation_analysis': correlation_data,
        'data_quality': data_quality,
        'sample_data': sample_data,
        'missing_values': df.isnull().sum().to_dict(),
        'unique_counts': {col: int(df[col].nunique()) for col in df.columns}
    }

def create_comprehensive_visualizations(df, chart_type, x_col=None, y_col=None, title="", color_col=None):
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
                                   f'Q-Q Plot of {y_col}', f'Density Plot of {y_col}'),
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
                    trendline="ols",
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

        CORRELATION ANALYSIS:
        {json.dumps(data_context['correlation_analysis'], indent=2)}

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
            }}
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
            }
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
            "primary_visualization": {"chart_type": None, "x_column": None, "y_column": None, "description": "No visualization available"}
        }

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
                            elif df[col].str.contains(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}', na=False).any():
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
            
            # Quick action buttons
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
                    "Are there any data quality issues I should address?"
                ]
                for question in sample_questions:
                    if st.button(f"🔸 {question}", key=f"sample_{hash(question)}"):
                        st.session_state.chat_query = question
            
            # Chat input
            user_query = st.text_input(
                "🎯 Ask me anything about your data:",
                value=getattr(st.session_state, 'chat_query', ''),
                placeholder="e.g., 'What are the strongest correlations in my data?' or 'Show me outliers in sales column'"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
            with col2:
                if st.button("🗑️ Clear Chat History", use_container_width=True):
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
                        
                        # Key insights
                        if insights.get('key_insights'):
                            st.markdown("**🔍 Key Insights:**")
                            for insight in insights['key_insights'][:5]:
                                st.markdown(f"• {insight}")
                        
                        # Numerical findings
                        if insights.get('numerical_findings'):
                            with st.expander("📊 Numerical Findings"):
                                for finding in insights['numerical_findings']:
                                    st.markdown(f"• {finding}")
                        
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
                                    st.plotly_chart(fig, use_container_width=True)
                        
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
            
            # Visualization controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                viz_type = st.selectbox(
                    "📈 Chart Type",
                    ["overview_dashboard", "correlation_heatmap", "distribution_analysis", 
                     "advanced_scatter", "categorical_analysis", "time_series", "multi_variable"],
                    format_func=lambda x: {
                        "overview_dashboard": "📊 Overview Dashboard",
                        "correlation_heatmap": "🔥 Correlation Heatmap", 
                        "distribution_analysis": "📈 Distribution Analysis",
                        "advanced_scatter": "🎯 Advanced Scatter Plot",
                        "categorical_analysis": "📋 Categorical Analysis",
                        "time_series": "📅 Time Series",
                        "multi_variable": "🌐 Multi-Variable Analysis"
                    }.get(x, x)
                )
            
            with col2:
                numeric_cols = data_info['basic_info']['numeric_columns']
                categorical_cols = data_info['basic_info']['categorical_columns']
                all_cols = df.columns.tolist()
                
                x_col = st.selectbox("🔤 X-Axis Column", [None] + all_cols, 
                                   index=1 if len(all_cols) > 0 else 0)
            
            with col3:
                y_col = st.selectbox("📊 Y-Axis Column", [None] + numeric_cols,
                                   index=1 if len(numeric_cols) > 0 else 0)
            
            # Color column (optional)
            color_col = st.selectbox("🎨 Color Column (Optional)", [None] + categorical_cols + numeric_cols[:3])
            
            # Generate visualization
            if st.button("🎨 Generate Visualization", type="primary"):
                with st.spinner("Creating visualization..."):
                    fig = create_comprehensive_visualizations(df, viz_type, x_col, y_col, 
                                                            f"{viz_type.replace('_', ' ').title()}", color_col)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Provide insights about the visualization
                        with st.expander("🔍 Visualization Insights"):
                            viz_query = f"Analyze this {viz_type.replace('_', ' ')} visualization showing {x_col} vs {y_col if y_col else 'data distribution'}. What patterns, trends, or insights can you identify?"
                            viz_insights = generate_comprehensive_insights_with_gemini(viz_query, data_info, model)
                            st.markdown(viz_insights.get('direct_answer', 'Analysis not available'))
            
            # Quick visualization grid
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
                            st.plotly_chart(fig_dist, use_container_width=True, key="dist1")
                
                with col2:
                    # Correlation heatmap if multiple numeric columns
                    if len(numeric_cols) > 1:
                        fig_corr = create_comprehensive_visualizations(df, 'correlation_heatmap')
                        if fig_corr:
                            st.plotly_chart(fig_corr, use_container_width=True, key="corr1")
            
            # Categorical analysis
            if categorical_cols:
                st.subheader("📋 Categorical Data Analysis")
                selected_cat = st.selectbox("Select categorical column:", categorical_cols)
                if selected_cat:
                    fig_cat = create_comprehensive_visualizations(df, 'categorical_analysis', 
                                                                selected_cat, None, 
                                                                f"Analysis of {selected_cat}")
                    if fig_cat:
                        st.plotly_chart(fig_cat, use_container_width=True)
        
        with tab3:
            st.header("📈 Statistical Analysis")
            
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
                
                # Statistical insights
                with st.expander("🧠 Statistical Insights"):
                    stats_query = "Provide detailed statistical insights about the numeric variables including distribution characteristics, skewness, outliers, and what these statistics tell us about the data"
                    stats_insights = generate_comprehensive_insights_with_gemini(stats_query, data_info, model)
                    st.markdown(stats_insights.get('direct_answer', 'Statistical analysis not available'))
            
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
                st.plotly_chart(fig_corr, use_container_width=True)
                
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
                                'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate'
                            })
                
                if strong_corr:
                    st.subheader("💪 Strong Correlations")
                    corr_df = pd.DataFrame(strong_corr).sort_values('Correlation', key=abs, ascending=False)
                    st.dataframe(corr_df, use_container_width=True)
            
            # Missing values analysis
            if data_info['missing_values']:
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
                                              labels=dict(x="Row Index", y="Columns"))
                        st.plotly_chart(fig_missing, use_container_width=True)
        
        with tab4:
            st.header("🔍 Data Explorer")
            
            # Data filtering
            st.subheader("🎯 Filter Data")
            
            # Create filters
            filters = {}
            
            # Numeric filters
            if numeric_cols:
                st.markdown("**📊 Numeric Filters:**")
                for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                    col_min, col_max = float(df[col].min()), float(df[col].max())
                    if col_min != col_max:
                        filters[col] = st.slider(
                            f"{col}",
                            min_value=col_min,
                            max_value=col_max,
                            value=(col_min, col_max),
                            step=(col_max - col_min) / 100
                        )
            
            # Categorical filters
            if categorical_cols:
                st.markdown("**📋 Categorical Filters:**")
                for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                    unique_values = df[col].dropna().unique()
                    if len(unique_values) <= 20:  # Only show filter if manageable number of categories
                        filters[col] = st.multiselect(
                            f"{col}",
                            options=unique_values,
                            default=unique_values
                        )
            
            # Apply filters
            filtered_df = df.copy()
            for col, filter_val in filters.items():
                if col in numeric_cols and isinstance(filter_val, tuple):
                    filtered_df = filtered_df[
                        (filtered_df[col] >= filter_val[0]) & 
                        (filtered_df[col] <= filter_val[1])
                    ]
                elif col in categorical_cols and filter_val:
                    filtered_df = filtered_df[filtered_df[col].isin(filter_val)]
            
            # Show filtered results
            st.subheader(f"📋 Filtered Data ({len(filtered_df)} rows)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(filtered_df):,}", f"{len(filtered_df) - len(df):,}")
            with col2:
                st.metric("% of Original", f"{len(filtered_df)/len(df)*100:.1f}%")
            with col3:
                st.metric("Filtered Out", f"{len(df) - len(filtered_df):,}")
            
            # Display filtered data
            st.dataframe(filtered_df, use_container_width=True)
            
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
        
        with tab5:
            st.header("📋 Comprehensive Data Report")
            
            # Generate comprehensive report
            if st.button("📊 Generate Full Report", type="primary"):
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
                    
                    # Key Statistics
                    if numeric_cols:
                        st.markdown("### 📈 Key Statistics")
                        summary_stats = df[numeric_cols].describe().round(2)
                        st.dataframe(summary_stats, use_container_width=True)
                    
                    # Visualizations
                    st.markdown("### 📊 Key Visualizations")
                    
                    # Overview dashboard
                    fig_overview = create_comprehensive_visualizations(df, 'overview_dashboard')
                    if fig_overview:
                        st.plotly_chart(fig_overview, use_container_width=True)
                    
                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    rec_query = "Based on this data analysis, provide specific recommendations for data improvement, further analysis, and business actions"
                    rec_insights = generate_comprehensive_insights_with_gemini(rec_query, data_info, model)
                    
                    recommendations = rec_insights.get('business_insights', [])
                    if recommendations:
                        for i, rec in enumerate(recommendations, 1):
                            st.markdown(f"{i}. {rec}")
                    
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
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Welcome to Smart Data Analysis Chat!
        
        This powerful tool combines AI-driven insights with interactive visualizations to help you understand your data better.
        
        ### ✨ Features:
        - 🤖 **AI-Powered Analysis**: Get intelligent insights using Google's Gemini AI
        - 📊 **Interactive Visualizations**: Create stunning charts and graphs
        - 📈 **Statistical Analysis**: Comprehensive statistical summaries and correlations
        - 🔍 **Data Explorer**: Filter and explore your data interactively  
        - 📋 **Automated Reports**: Generate professional data analysis reports
        - 💬 **Natural Language Queries**: Ask questions about your data in plain English
        
        ### 🏁 Getting Started:
        1. **Upload your CSV file** using the sidebar
        2. **Explore your data** using the interactive tabs
        3. **Ask questions** using natural language in the AI Chat tab
        4. **Create visualizations** to better understand patterns
        5. **Generate reports** for sharing insights
        
        ### 💡 Example Questions You Can Ask:
        - "What are the main trends in my sales data?"
        - "Which factors are most correlated with customer satisfaction?"
        - "Are there any outliers in my revenue data?"
        - "What's the distribution of my product categories?"
        - "Can you identify any seasonal patterns?"
        
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
            """)

if __name__ == "__main__":
    main()