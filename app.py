import os
import json
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    gemini_status = "Connected"
except Exception as e:
    gemini_model = None
    gemini_status = f"Error: {str(e)}"

# Initialize Spark (Optional)
try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("DataLakeAnalytics") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark_status = "Connected"
except Exception as e:
    spark = None
    spark_status = "Not Connected"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_visualization(data, chart_type, title, filename):
    """Create and save matplotlib visualizations."""
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']

    if chart_type == 'bar':
        bars = plt.bar(data['labels'], data['values'], color=colors[0], alpha=0.85, edgecolor='white', linewidth=1.5)
        plt.xlabel(data.get('xlabel', 'Categories'), fontsize=12, fontweight='bold')
        plt.ylabel(data.get('ylabel', 'Values'), fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
                    
    elif chart_type == 'pie':
        plt.pie(data['values'], labels=data['labels'], autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
        plt.axis('equal')
        
    elif chart_type == 'heatmap':
        sns.heatmap(data['matrix'], annot=True, fmt='.2f', cmap='RdYlBu_r',
                    xticklabels=data['labels'], yticklabels=data['labels'],
                    cbar_kws={'label': 'Correlation'}, linewidths=0.5)

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    return filepath

def get_data_context(filepath, file_extension):
    """Extract comprehensive context from the dataset."""
    try:
        if file_extension == 'csv':
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Context string generation for AI
        buffer = []
        buffer.append(f"Dataset Overview:")
        buffer.append(f"- Columns: {', '.join(df.columns.tolist())}")
        buffer.append("\nData Types:")
        for col, dtype in df.dtypes.items():
            buffer.append(f"  {col}: {dtype}")
        buffer.append("\nFirst 3 Rows:")
        buffer.append(df.head(3).to_string())
        
        return "\n".join(buffer), df
    except Exception as e:
        return f"Error reading data: {str(e)}", None

def run_normal_analysis(context, prompt):
    """Helper function for normal text analysis"""
    full_prompt = f"""
    You are an expert Data Analyst.
    Dataset Context:
    {context}
    
    User Request: "{prompt}"
    
    Provide a detailed, professional analysis formatted in Markdown.
    Use bullet points and bold text for readability.
    """
    response = gemini_model.generate_content(full_prompt)
    return response.text

@app.route('/')
def index():
    return render_template('index.html', spark_status=spark_status, gemini_status=gemini_status)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type.'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        file_size = os.path.getsize(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'extension': filename.rsplit('.', 1)[1].lower(),
            'size': f"{file_size / 1024:.2f} KB"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Run AI-powered analysis or Data Filtering."""
    try:
        data = request.json
        filepath = data.get('filepath')
        file_extension = data.get('extension')
        user_prompt = data.get('prompt', 'Provide a comprehensive analysis.')

        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found on server'}), 404

        # Get Data
        context_str, df = get_data_context(filepath, file_extension)
        
        if df is None:
            return jsonify({'success': False, 'error': context_str}), 500

        # --- Generate Charts (Standard Logic) ---
        charts = []
        numeric_df = df.select_dtypes(include=[np.number])
        
        # 1. Null Values Chart
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            null_data = {
                'labels': [str(x) for x in null_counts[null_counts > 0].index],
                'values': list(null_counts[null_counts > 0].values),
                'xlabel': 'Columns', 'ylabel': 'Missing Values'
            }
            chart_path = create_visualization(null_data, 'bar', 'Missing Values', f'nulls_{datetime.now().timestamp()}.png')
            charts.append({'title': 'Missing Values', 'file': os.path.basename(chart_path), 'description': 'Missing data counts'})

        # 2. Distribution (First numeric column)
        if not numeric_df.empty:
            first_numeric = numeric_df.columns[0]
            value_counts = df[first_numeric].value_counts().head(10)
            dist_data = {
                'labels': [str(x) for x in value_counts.index],
                'values': list(value_counts.values),
                'xlabel': first_numeric, 'ylabel': 'Count'
            }
            chart_path = create_visualization(dist_data, 'bar', f'{first_numeric} Distribution', f'dist_{datetime.now().timestamp()}.png')
            charts.append({'title': f'{first_numeric} Distribution', 'file': os.path.basename(chart_path), 'description': 'Top 10 values'})

        # --- AI INTELLIGENT FILTERING LOGIC ---
        ai_response = ""
        
        if gemini_model:
            columns_list = ", ".join(df.columns.tolist())
            
            # Ask Gemini: Is this a filter request or analysis request?
       # Update this part inside app.py

            filter_prompt = f"""
            You are a Python Pandas Data Expert.
            Dataset Columns: [{columns_list}]
            
            User Query: "{user_prompt}"
            
            INSTRUCTIONS:
            1. If the user is asking to FILTER, LIST, or SHOW rows:
               - Write a VALID SINGLE LINE pandas filtering syntax.
               - Use variable 'df'.
               - IMPORTANT: Use .str.contains() for text matching to find partial matches.
               - Example: df[df['issue_type'].str.contains('Payment', case=False, na=False)]
               - Return ONLY the code string. No markdown.
            
            2. If the user asking for summary/analysis:
               - Return exactly: "ANALYSIS_MODE"
            """

            response = gemini_model.generate_content(filter_prompt)
            ai_decision = response.text.strip().replace('`', '').replace('python', '').strip()
            print(f"AI Decision: {ai_decision}") # Debug print

            if "ANALYSIS_MODE" not in ai_decision and "df" in ai_decision:
                try:
                    # Execute filter code
                    local_vars = {'df': df}
                    # Safe execution wrapper
                    filtered_result = eval(ai_decision, {"__builtins__": {}}, local_vars)
                    if isinstance(filtered_result, pd.Series):
                        filtered_result = filtered_result.to_frame()
                    
                    if isinstance(filtered_result, pd.DataFrame):
                        rows_count = len(filtered_result)
                        if rows_count > 0:
                            # Convert to styled HTML table
                            table_html = filtered_result.to_html(
                                classes='table custom-table', 
                                index=False,
                                border=0,
                                na_rep='-'
                            )
                            ai_response = f"""
                            <h3>🎯 Filtered Data Results</h3>
                            <p>I found <strong>{rows_count}</strong> rows matching your criteria: <code>{user_prompt}</code></p>
                            <div class="table-responsive-custom">
                                {table_html}
                            </div>
                            """
                        else:
                            ai_response = "### ⚠️ No Results\nNo rows matched your filter criteria."
                    else:
                        # If result is a single value (like count/sum)
                        ai_response = f"### Result\n**Value:** {filtered_result}"
                        
                except Exception as e:
                    print(f"Filter execution failed: {e}")
                    ai_response = run_normal_analysis(context_str, user_prompt)
            else:
                # Normal Analysis
                ai_response = run_normal_analysis(context_str, user_prompt)

        else:
             ai_response = "AI Service Unavailable."

        return jsonify({
            'success': True,
            'ai_response': ai_response,
            'charts': charts,
            'summary': {
                'rows': int(len(df)),
                'columns': int(len(df.columns)),
                'numeric_columns': int(len(numeric_df.columns)),
                'missing_cells': int(df.isnull().sum().sum())
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/download/<filename>')
def download(filename):
    try:
        return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)
    except Exception:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)