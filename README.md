🧠 DataLake AI Analyst
DataLake AI Analyst is an intelligent, web-based data analytics platform. It allows users to upload datasets (CSV/Excel) and interact with them using Natural Language.

Powered by Google Gemini AI and Python Pandas, it translates plain English questions (e.g., "List customers with payment failures") into code, filters the data instantly, and renders it in a beautiful, interactive table format. It also automatically generates statistical visualizations.

🚀 Features
📂 Drag & Drop Upload: Supports .csv, .xlsx, and .xls files (up to 100MB).

💬 Natural Language Querying: Ask questions like "Show me rows where Status is Pending" or "List users from Chennai".

🤖 AI-Powered Filtering: automatically converts text prompts into Python Pandas filtering logic using Google Gemini.

📊 Smart Visualizations: Auto-generates Bar charts, Pie charts, and Correlation Heatmaps.

📉 Interactive Tables: Results are displayed in a responsive, auto-expanding table with sticky headers.

📥 Export Results: Download generated charts and reports.

⚡ Hybrid Processing: Built with Flask and capable of integrating Apache Spark (optional) for big data.

🛠️ Tech Stack
Frontend: HTML5, CSS3 (Responsive Grid), JavaScript (Fetch API).

Backend: Python, Flask.

Data Processing: Pandas, NumPy.

AI Engine: Google Generative AI (Gemini Pro/Flash).

Visualization: Matplotlib, Seaborn.

Big Data (Optional): Apache Spark (PySpark).

⚙️ Installation Guide
Follow these steps to run the project locally.

1. Clone the Repository
Bash

[git clone https://github.com/yourusername/datalake-ai.git
cd datalake-ai](https://github.com/sarjanthecoder/BIG-DATA-ANALYZER.git)
2. Set up a Virtual Environment (Recommended)
Bash

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Create a file named requirements.txt (content provided below) and run:

Bash

pip install -r requirements.txt
4. Configure API Key
Create a file named .env in the root directory and add your Google Gemini API key:

Code snippet

GEMINI_API_KEY=your_actual_api_key_here
5. Run the Application
Bash

python app.py
You will see output indicating the server is running (e.g., Running on http://127.0.0.1:5000).

📖 Usage
Open your browser and go to http://127.0.0.1:5000.

Upload Data: Drag and drop your CSV or Excel file into the upload zone.

Ask a Question:

Filter Example: "List details of customers who have Payment Issue"

Analysis Example: "Give me a summary of this dataset"

View Results:

If you asked for a filter, a Table will appear.

If you asked for analysis, a Report will appear.

Scroll down to see auto-generated Charts.

📂 Project Structure
Plaintext

DataLake-AI/
├── app.py                 # Main Flask Backend & AI Logic
├── .env                   # API Keys (Not uploaded to GitHub)
├── requirements.txt       # Python Dependencies
├── uploads/               # Temp folder for uploaded files
├── outputs/               # Temp folder for generated charts
└── templates/
    └── index.html         # Frontend UI with CSS/JS
🐛 Troubleshooting
"AI Service Unavailable": Check if your GEMINI_API_KEY in the .env file is correct and active.

"No file selected": Ensure you are uploading a valid CSV or Excel file.

Table not showing (Text result only): Ensure you are using the updated app.py logic that converts pd.Series to pd.DataFrame.



📜 License
This project is open-source and available under the MIT License.

✅ dependencies (requirements.txt)
Create a file named requirements.txt and paste this inside:

Plaintext

Flask
pandas
numpy
matplotlib
seaborn
google-generativeai
python-dotenv
openpyxl
werkzeug
