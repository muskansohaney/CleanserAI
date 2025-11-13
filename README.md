<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/2936/2936635.png" width="90" />
</p>

<h1 align="center">🧼 CleanserAI — Your Intelligent Data Cleaning Companion</h1>

<p align="center">
  <b>Transforming data preprocessing from hours to minutes 🚀</b><br>
  An AI-powered Streamlit app that automates messy data handling, profiling, and preprocessing in one clean interface.
</p>

---

## ✨ Features

✅ **Missing Value Detection & Imputation**  
Automatically finds and fills missing data using mean, median, mode, or constant strategies.

✅ **Outlier Detection**  
Detect anomalies using IQR or advanced models like IsolationForest (PyOD).

✅ **Duplicate Handling**  
Identify and remove duplicate rows seamlessly.

✅ **Zero-to-NaN Conversion**  
Turn invalid zeros into missing values for proper treatment.

✅ **Categorical Encoding**  
Supports Label, Ordinal, and OneHot encoding methods.

✅ **Feature Scaling & Normalization**  
Choose between Standard, MinMax, or Robust scaling.

✅ **Correlation Analysis & Feature Pruning**  
Visualize relationships and auto-drop highly correlated columns.

✅ **Automated EDA / Profiling**  
Generate full interactive reports using `ydata-profiling`.

✅ **Export Options**  
Download the cleaned dataset or export preprocessing pipelines.

---

## 🧠 Tech Stack

| Component | Library |
|------------|----------|
| Web UI | [Streamlit](https://streamlit.io) |
| Data Processing | Pandas, NumPy |
| ML Preprocessing | Scikit-learn |
| Profiling | ydata-profiling, streamlit-pandas-profiling |
| Outlier Detection | PyOD |
| Visualization | Plotly |

---

## 🖥️ Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/muskansohaney/CleanserAI.git
cd CleanserAI
```
### 2️⃣ Create and activate a virtual environment
```bash
python3 -m venv .venv   
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Launch the app
```bash
streamlit run app.py
```   
Then open your browser at http://localhost:8501   

---

###  🧩 Folder Structure   
CleanserAI/   
│   
├── app.py                     # Main Streamlit app   
├── requirements.txt            # All dependencies   
├── .streamlit/   
│   └── config.toml             # UI theme settings   
├── README.md                   # You’re reading this 🙂   
├── .gitignore   
└── demo.png                    # App preview (add screenshot here)   

---

### 🏗️ Future Enhancements   
🔮 Auto feature selection using SHAP   
☁️ Cloud dataset upload via Google Drive / S3   
📈 Model readiness scoring   
🧬 Deep anomaly detection using AutoEncoders   


Developed by Muskan Sohaney
---
