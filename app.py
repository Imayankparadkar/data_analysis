import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import os
import json
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
    # Try to get API key from environment variables (Render environment variables)
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        # If not found in environment, try Streamlit secrets (fallback)
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except:
            st.error("❌ GEMINI_API_KEY not found. Please set it in your Render environment variables.")
            st.info("🔧 In Render dashboard, go to Environment → Add Environment Variable → GEMINI_API_KEY")
            st.stop()
    
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"❌ Error initializing Gemini API: {str(e)}")
        st.info("Please check your API key is valid and try again.")
        st.stop()

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
           "data_quality_assessment": ["Analysis could not be completed"],
            "visualization_recommendations": [],
            "business_insights": ["Please check your data and try again"],
            "follow_up_questions": ["What type of analysis would you like to perform?"],
            "primary_visualization": {
                "chart_type": "overview_dashboard",
                "x_column": None,
                "y_column": None,
                "description": "Basic overview"
            }
        }

def display_insights_ui(insights):
    """Display insights in an organized UI"""
    
    # Direct answer in a prominent box
    st.markdown(f"""
    <div class="insight-box">
        <h3>🎯 Analysis Result</h3>
        <p>{insights.get('direct_answer', 'No direct answer available')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different types of insights
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Key Insights", 
        "📊 Numbers & Stats", 
        "📈 Visualizations", 
        "💼 Business Impact", 
        "❓ Follow-up"
    ])
    
    with tab1:
        st.subheader("Key Insights")
        for i, insight in enumerate(insights.get('key_insights', []), 1):
            st.markdown(f"**{i}.** {insight}")
        
        if 'statistical_summary' in insights:
            st.subheader("Statistical Summary")
            for stat in insights['statistical_summary']:
                st.markdown(f"• {stat}")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Numerical Findings")
            for finding in insights.get('numerical_findings', []):
                st.markdown(f"📈 {finding}")
        
        with col2:
            st.subheader("Data Quality")
            for quality in insights.get('data_quality_assessment', []):
                st.markdown(f"🔍 {quality}")
    
    with tab3:
        st.subheader("Recommended Visualizations")
        viz_recs = insights.get('visualization_recommendations', [])
        
        if viz_recs:
            for i, viz in enumerate(viz_recs):
                with st.expander(f"📊 {viz.get('chart_type', 'Chart').replace('_', ' ').title()} - {viz.get('priority', 'medium').upper()} Priority"):
                    st.write(f"**Description:** {viz.get('description', 'No description')}")
                    if viz.get('x_column'):
                        st.write(f"**X-axis:** {viz['x_column']}")
                    if viz.get('y_column'):
                        st.write(f"**Y-axis:** {viz['y_column']}")
                    if viz.get('color_column'):
                        st.write(f"**Color by:** {viz['color_column']}")
                    
                    # Create visualization button
                    if st.button(f"Create {viz.get('chart_type', 'chart').replace('_', ' ').title()}", key=f"viz_{i}"):
                        create_and_display_chart(viz)
        else:
            st.info("No specific visualization recommendations available.")
    
    with tab4:
        st.subheader("Business Insights")
        business_insights = insights.get('business_insights', [])
        if business_insights:
            for insight in business_insights:
                st.markdown(f"💡 {insight}")
        else:
            st.info("No business insights available for this analysis.")
    
    with tab5:
        st.subheader("Suggested Follow-up Questions")
        follow_ups = insights.get('follow_up_questions', [])
        if follow_ups:
            for i, question in enumerate(follow_ups):
                if st.button(f"❓ {question}", key=f"followup_{i}"):
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    st.rerun()
        else:
            st.info("No follow-up questions available.")

def create_and_display_chart(viz_config):
    """Create and display chart based on configuration"""
    if st.session_state.current_data is not None:
        chart = create_comprehensive_visualizations(
            st.session_state.current_data,
            viz_config.get('chart_type', 'overview_dashboard'),
            viz_config.get('x_column'),
            viz_config.get('y_column'),
            viz_config.get('description', ''),
            viz_config.get('color_column')
        )
        
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.error("Could not create the requested visualization.")

def main():
    # Initialize
    initialize_session_state()
    model = initialize_gemini()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Smart Data Analysis Chat</h1>
        <p>Upload your data and chat with AI to get instant insights and visualizations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📂 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file to start analyzing your data"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                st.session_state.current_data = df
                
                # Analyze data
                with st.spinner("Analyzing your data..."):
                    st.session_state.data_info = comprehensive_data_analysis(df)
                
                st.success(f"✅ Data loaded successfully!")
                
                # Display basic info
                st.markdown(f"""
                <div class="metric-container">
                    <h4>📊 Dataset Overview</h4>
                    <p><strong>Rows:</strong> {df.shape[0]:,}</p>
                    <p><strong>Columns:</strong> {df.shape[1]}</p>
                    <p><strong>Completeness:</strong> {st.session_state.data_info['data_quality']['completeness_score']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        # Quick stats if data is loaded
        if st.session_state.current_data is not None:
            st.header("🔍 Quick Stats")
            
            data_info = st.session_state.data_info
            
            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", f"{data_info['basic_info']['shape'][0]:,}")
                st.metric("Numeric Cols", len(data_info['basic_info']['numeric_columns']))
            
            with col2:
                st.metric("Total Columns", data_info['basic_info']['shape'][1])
                st.metric("Text Cols", len(data_info['basic_info']['categorical_columns']))
            
            # Data quality indicators
            st.subheader("Data Quality")
            completeness = data_info['data_quality']['completeness_score']
            
            if completeness >= 90:
                st.success(f"Excellent: {completeness:.1f}% complete")
            elif completeness >= 70:
                st.warning(f"Good: {completeness:.1f}% complete")
            else:
                st.error(f"Needs attention: {completeness:.1f}% complete")
            
            duplicates = data_info['data_quality']['duplicate_percentage']
            if duplicates > 0:
                st.info(f"📋 {duplicates:.1f}% duplicate rows")
    
    # Main content area
    if st.session_state.current_data is not None:
        # Chat interface
        st.header("💬 Chat with Your Data")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and "insights" in message:
                    display_insights_ui(message["insights"])
                else:
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask anything about your data..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    insights = generate_comprehensive_insights_with_gemini(
                        prompt, 
                        st.session_state.data_info, 
                        model
                    )
                    
                    # Display insights
                    display_insights_ui(insights)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": insights.get('direct_answer', ''),
                        "insights": insights
                    })
                    
                    # Auto-generate primary visualization
                    if 'primary_visualization' in insights:
                        primary_viz = insights['primary_visualization']
                        if primary_viz.get('chart_type'):
                            st.subheader("📊 Primary Visualization")
                            chart = create_comprehensive_visualizations(
                                st.session_state.current_data,
                                primary_viz.get('chart_type'),
                                primary_viz.get('x_column'),
                                primary_viz.get('y_column'),
                                primary_viz.get('description', ''),
                                None
                            )
                            if chart:
                                st.plotly_chart(chart, use_container_width=True)
        
        # Quick action buttons
        st.header("🚀 Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Create Dashboard", use_container_width=True):
                chart = create_comprehensive_visualizations(
                    st.session_state.current_data, 
                    'overview_dashboard', 
                    title="Data Overview Dashboard"
                )
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
        
        with col2:
            if st.button("🔗 Show Correlations", use_container_width=True):
                numeric_cols = st.session_state.data_info['basic_info']['numeric_columns']
                if len(numeric_cols) > 1:
                    chart = create_comprehensive_visualizations(
                        st.session_state.current_data, 
                        'correlation_heatmap',
                        title="Correlation Analysis"
                    )
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for correlation analysis")
        
        with col3:
            if st.button("📈 Distribution Analysis", use_container_width=True):
                numeric_cols = st.session_state.data_info['basic_info']['numeric_columns']
                if numeric_cols:
                    chart = create_comprehensive_visualizations(
                        st.session_state.current_data, 
                        'distribution_analysis',
                        y_col=numeric_cols[0],
                        title=f"Distribution of {numeric_cols[0]}"
                    )
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                else:
                    st.warning("No numeric columns found for distribution analysis")
        
        with col4:
            if st.button("🎯 Data Summary", use_container_width=True):
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": "Give me a comprehensive summary of my dataset"
                })
                st.rerun()
        
        # Data preview
        with st.expander("👀 Data Preview", expanded=False):
            st.subheader("First 10 rows of your data:")
            st.dataframe(st.session_state.current_data.head(10), use_container_width=True)
            
            st.subheader("Dataset Info:")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Column Information:**")
                info_df = pd.DataFrame({
                    'Column': st.session_state.current_data.columns,
                    'Type': st.session_state.current_data.dtypes.astype(str),
                    'Non-Null': st.session_state.current_data.count(),
                    'Null Count': st.session_state.current_data.isnull().sum()
                })
                st.dataframe(info_df, use_container_width=True)
            
            with col2:
                st.write("**Missing Values:**")
                missing_df = pd.DataFrame({
                    'Column': st.session_state.current_data.columns,
                    'Missing Count': st.session_state.current_data.isnull().sum(),
                    'Missing %': (st.session_state.current_data.isnull().sum() / len(st.session_state.current_data) * 100).round(2)
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
                if not missing_df.empty:
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("No missing values found! 🎉")
    
    else:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to Smart Data Analysis Chat!
        
        This AI-powered tool helps you analyze your data through natural language conversations.
        
        ### 🚀 Getting Started:
        1. **Upload your CSV file** using the sidebar
        2. **Ask questions** about your data in plain English
        3. **Get instant insights** with visualizations and analysis
        4. **Explore deeper** with follow-up questions
        
        ### 💡 Example Questions:
        - "What are the main patterns in my data?"
        - "Show me the correlation between variables"
        - "Which columns have missing values?"
        - "Create a dashboard for my data"
        - "Find outliers in the numeric columns"
        - "What's the distribution of [column name]?"
        
        ### 📊 Features:
        - **Interactive Charts**: Correlation heatmaps, distributions, scatter plots
        - **Statistical Analysis**: Comprehensive stats, outlier detection
        - **Data Quality Assessment**: Missing values, duplicates, data types
        - **Smart Recommendations**: Suggested visualizations and analyses
        - **Business Insights**: Actionable findings from your data
        
        **Ready to start?** Upload your CSV file in the sidebar! 📂
        """)
        
        # Sample data section
        st.markdown("---")
        st.subheader("🎯 Don't have data? Try our sample!")
        
        if st.button("Generate Sample Dataset", use_container_width=True):
            # Create sample data
            np.random.seed(42)
            sample_data = pd.DataFrame({
                'sales': np.random.normal(1000, 200, 100),
                'marketing_spend': np.random.normal(500, 100, 100),
                'customer_satisfaction': np.random.uniform(1, 5, 100),
                'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
                'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 100),
                'month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], 100)
            })
            
            # Add some correlations
            sample_data['sales'] = sample_data['sales'] + sample_data['marketing_spend'] * 0.5 + np.random.normal(0, 50, 100)
            sample_data['profit'] = sample_data['sales'] * 0.3 + np.random.normal(0, 50, 100)
            
            st.session_state.current_data = sample_data
            st.session_state.data_info = comprehensive_data_analysis(sample_data)
            
            st.success("✅ Sample dataset loaded! You can now start chatting with your data.")
            st.rerun()

if __name__ == "__main__":
    main()