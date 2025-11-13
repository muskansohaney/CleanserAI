# ⚙️ CleanserAI– AI-Powered Data Preprocessing Studio

CleanserAI automates data cleaning, preprocessing, and profiling in minutes — not hours.

### 🚀 Features
- Missing value detection & imputation
- Outlier detection (IQR + IsolationForest)
- Categorical encoding (Label, Ordinal, OneHot)
- Feature scaling & normalization
- Correlation heatmap & feature pruning
- Automated EDA with ydata-profiling
- Duplicate handling & winsorization
- Export cleaned dataset & preprocessing pipeline

### 🧠 Tech Stack
Python | Streamlit | Pandas | NumPy | Scikit-learn | Plotly | PyOD | ydata-profiling

### 🧩 How to Run Locally
```bash
git clone https://github.com/muskansohaney/DataForge.git
cd DataForge
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
