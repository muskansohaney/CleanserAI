# CleanserAI – Advanced UI with consistent state (fixed profiling / state sync)
import io
import pickle
import base64
from typing import Tuple, List, Dict

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OrdinalEncoder, OneHotEncoder, LabelEncoder,
    StandardScaler, MinMaxScaler, RobustScaler
)
from sklearn.compose import ColumnTransformer

# Optional dependencies flags
HAS_PROFILING = False
HAS_PYOD = False

try:
    from ydata_profiling import ProfileReport
    from streamlit_pandas_profiling import st_profile_report
    HAS_PROFILING = True
except Exception:
    pass

try:
    from pyod.models.iforest import IForest
    HAS_PYOD = True
except Exception:
    pass

# Page config & styling
st.set_page_config(page_title="CleanserAI", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #f5f6fa; border-right: 2px solid #e3e3e3; }
    div[data-testid="stMetricValue"] { color: #635BFF; font-weight: 600; }
    .block-container { padding-top: 1rem; }
    h1, h2, h3 { color: #2c2c2c; }
</style>
""", unsafe_allow_html=True)

from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header

st.markdown("<h1 style='text-align:center; color:#635BFF;'>CleanserAI – Data Preprocessing Studio</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Transforming data cleaning from hours to minutes</p>", unsafe_allow_html=True)
colored_header(label="", description="", color_name="violet-70")

# Sidebar & navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2936/2936635.png", width=100)
    st.header(" Navigation")

    selected = option_menu(
        menu_title=None,
        options=["Overview", "Missing Values", "Outliers", "Encoding & Scaling", "Correlation", "Profiling", "Export"],
        icons=["table", "droplet", "activity", "sliders", "bar-chart", "book", "download"],
        menu_icon="cast",
        default_index=0,
    )

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV / XLSX / JSON", type=["csv", "xlsx", "xls", "json"])

# Helpers
@st.cache_data
def read_file(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith((".xls", ".xlsx")):
        return pd.read_excel(file)
    elif name.endswith(".json"):
        return pd.read_json(file)
    else:
        return pd.read_csv(file)


def detect_numeric_cat(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    if df is None or df.empty:
        return [], []
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return num, cat


def df_overview(df: pd.DataFrame) -> Dict:
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "memory_mb": df.memory_usage(deep=True).sum() / 1024 ** 2,
        "dtypes": df.dtypes.astype(str).to_dict()
    }


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isnull().sum().rename("missing_count")
    miss_pct = (df.isnull().mean() * 100).rename("missing_pct")
    return pd.concat([miss, miss_pct], axis=1).sort_values("missing_pct", ascending=False)


def impute_dataframe(df, num_strategy="mean", cat_strategy="most_frequent"):
    num_cols, cat_cols = detect_numeric_cat(df)
    num_imp = SimpleImputer(strategy=num_strategy)
    cat_imp = SimpleImputer(strategy=cat_strategy)
    preprocessor = ColumnTransformer([
        ("num", num_imp, num_cols),
        ("cat", cat_imp, cat_cols)
    ], remainder="passthrough")
    arr = preprocessor.fit_transform(df)
    cols = num_cols + cat_cols
    out = pd.DataFrame(arr, columns=cols, index=df.index)
    for c in num_cols:
        out[c] = pd.to_numeric(out[c])
    return out, preprocessor


def iqr_outlier_mask(series, k=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (series < lower) | (series > upper)


def remove_outliers_iqr(df, columns, k=1.5):
    mask = pd.Series(False, index=df.index)
    for c in columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            mask |= iqr_outlier_mask(df[c], k)
    return df.loc[~mask].copy()


def winsorize_series(series, lower_q=0.05, upper_q=0.95):
    low = series.quantile(lower_q)
    high = series.quantile(upper_q)
    return series.clip(low, high)


def apply_encoding(df, categorical_cols: List[str], encoding: str = "onehot"):
    df_enc = df.copy()
    fitted = None
    if encoding == "label":
        for col in categorical_cols:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        fitted = "label"
    elif encoding == "ordinal":
        oe = OrdinalEncoder()
        df_enc[categorical_cols] = oe.fit_transform(df_enc[categorical_cols].astype(str))
        fitted = oe
    elif encoding == "onehot":
        df_enc = pd.get_dummies(df_enc, columns=categorical_cols, drop_first=False)
        fitted = "onehot"
    return df_enc, fitted


def apply_scaling(df, numeric_cols: List[str], scaler_name: str = "standard"):
    df_scaled = df.copy()
    scaler = None
    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()
    else:
        return df_scaled, None
    # If numeric cols exist in df (after encoding they might change), check presence
    numeric_now = [c for c in numeric_cols if c in df_scaled.columns]
    if numeric_now:
        df_scaled[numeric_now] = scaler.fit_transform(df_scaled[numeric_now])
    return df_scaled, scaler


def correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str]):
    if not numeric_cols:
        return None
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix")
    return fig


def get_download_link(df, filename="cleaned_data.csv"):
    towrite = io.BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    return f"data:file/csv;base64,{b64}"


# Main app logic
if uploaded_file:
    # Load and initialize working_df from session state if present
    original_df = read_file(uploaded_file)
    if "df" not in st.session_state:
        st.session_state["df"] = original_df.copy()
    # working_df is the single source of truth used everywhere
    working_df = st.session_state.get("df", original_df.copy())

    num_cols, cat_cols = detect_numeric_cat(working_df)

    # Overview
    if selected == "Overview":
        st.subheader("Dataset Overview")
        ov = df_overview(working_df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", ov["rows"])
        c2.metric("Columns", ov["columns"])
        c3.metric("Memory (MB)", f"{ov['memory_mb']:.2f}")
        st.markdown("**Preview**")
        st.dataframe(working_df.head(10), use_container_width=True)
        st.markdown("**Data types**")
        st.dataframe(pd.DataFrame.from_dict(ov["dtypes"], orient="index", columns=["dtype"]).reset_index().rename(columns={"index": "column"}), use_container_width=True)

    # Missing Values
    elif selected == "Missing Values":
        st.subheader("🧩 Missing Value Treatment")
        mv = missing_value_summary(working_df)
        st.dataframe(mv, use_container_width=True)
        if mv["missing_count"].sum() > 0:
            num_strategy = st.selectbox("Numerical imputation strategy", ["mean", "median", "most_frequent", "constant"])
            cat_strategy = st.selectbox("Categorical imputation strategy", ["most_frequent", "constant"])
            if num_strategy == "constant":
                const_num = st.number_input("Numeric constant value", value=0.0)
            else:
                const_num = None
            if cat_strategy == "constant":
                const_cat = st.text_input("Categorical constant value", value="missing")
            else:
                const_cat = None

            if st.button("Apply Imputation"):
                # Optional: convert 0s to NaN for numeric columns where 0 is invalid
                cols_to_clean = st.multiselect("Select columns where 0 means missing:", working_df.columns.tolist())
                if st.button("Convert 0s to NaN in selected columns"):
                    for c in cols_to_clean:
                        working_df.loc[working_df[c] == 0, c] = np.nan
                    st.session_state["df"] = working_df
                    st.success(f"Converted zeros to NaN in columns: {cols_to_clean}")
                    st.rerun()

                # map 'constant' to SimpleImputer behavior if needed
                n_strategy = "constant" if num_strategy == "constant" else num_strategy
                c_strategy = "constant" if cat_strategy == "constant" else cat_strategy
                imputed_df, preproc = impute_dataframe(working_df, num_strategy=n_strategy, cat_strategy=c_strategy)
                # If constant used, replace numeric/cat columns with chosen values post-imputation
                if num_strategy == "constant" and const_num is not None:
                    num_cols_now, _ = detect_numeric_cat(imputed_df)
                    for c in num_cols_now:
                        imputed_df[c] = imputed_df[c].fillna(const_num)
                if cat_strategy == "constant" and const_cat is not None:
                    _, cat_cols_now = detect_numeric_cat(imputed_df)
                    # above detect returns numeric then categorical; reuse original cat_cols
                    for c in cat_cols:
                        if c in imputed_df.columns:
                            imputed_df[c] = imputed_df[c].fillna(const_cat)
                # Save back to session
                st.session_state["df"] = imputed_df
                working_df = imputed_df
                st.success("Imputation applied")
                st.dataframe(working_df.head(10), use_container_width=True)
        else:
            st.info("No missing values detected.")
        st.markdown("### Handle Zero Values")
        zero_cols = st.multiselect(
            "Select numeric columns where 0 should be treated as missing:",
            working_df.select_dtypes(include=[np.number]).columns.tolist()
        )

        if st.button("Convert 0s to NaN"):
            for col in zero_cols:
                zero_count = (working_df[col] == 0).sum()
                working_df.loc[working_df[col] == 0, col] = np.nan
                st.write(f"Converted {zero_count} zero values to NaN in column '{col}'")
            st.session_state["df"] = working_df
            st.success("Zero values converted to NaN. Now you can re-run imputation.")
            st.rerun()

    # Outliers
    elif selected == "Outliers":
        st.subheader("Outlier Detection & Handling")
        num_cols, cat_cols = detect_numeric_cat(working_df)
        st.write(f"Numeric columns: {num_cols}")
        outlier_method = st.selectbox("Outlier detection method", options=["IQR (default)"] + (["IsolationForest (PyOD)"] if HAS_PYOD else []))
        if outlier_method.startswith("IQR"):
            k = st.slider("IQR multiplier (k)", min_value=1.0, max_value=4.0, value=1.5)
            if st.button("Remove outliers (IQR)"):
                before = working_df.shape[0]
                cleaned = remove_outliers_iqr(working_df, num_cols, k=k)
                st.session_state["df"] = cleaned
                working_df = cleaned
                after = cleaned.shape[0]
                st.success(f"Removed {before - after} rows identified as outliers")
                st.dataframe(working_df.head(10), use_container_width=True)
        else:
            if not HAS_PYOD:
                st.warning("PyOD not installed. Install `pyod` to use IsolationForest or other algorithms.")
            else:
                contamination = st.slider("contamination (fraction outliers)", 0.01, 0.5, 0.05)
                if st.button("Detect & remove with IsolationForest"):
                    model = IForest(contamination=contamination)
                    X = working_df[num_cols].fillna(0).values
                    model.fit(X)
                    preds = model.labels_
                    mask = preds == 0
                    cleaned = working_df.loc[mask].copy()
                    st.session_state["df"] = cleaned
                    working_df = cleaned
                    st.success(f"Removed {(~mask).sum()} rows marked as outliers by IForest")
                    st.dataframe(working_df.head(10), use_container_width=True)

        # Winsorize
        st.markdown("---")
        if st.button("Apply Winsorize (5th-95th)"):
            dfw = working_df.copy()
            num_cols_now, _ = detect_numeric_cat(dfw)
            for col in num_cols_now:
                if pd.api.types.is_numeric_dtype(dfw[col]):
                    dfw[col] = winsorize_series(dfw[col], 0.05, 0.95)
            st.session_state["df"] = dfw
            working_df = dfw
            st.success("Applied winsorization to numeric columns")
            st.dataframe(working_df.head(10), use_container_width=True)

        # -------------------- Duplicate Removal UI (copy into your app) --------------------
        st.markdown("---")
        st.subheader("🧹 Duplicate Records (Preview & Remove)")

        # Use the working dataframe (single source of truth)
        working_df = st.session_state.get("df", None)
        if working_df is None:
            st.info("No dataset loaded yet.")
        else:
            # Show number of exact duplicate rows (counting all repeated rows)
            dup_mask_all = working_df.duplicated(keep=False)          # True for every row that has a duplicate
            total_dups = int(dup_mask_all.sum())
            st.write(f"Total rows: **{working_df.shape[0]}** — Duplicate rows (all occurrences): **{total_dups}**")

            # Let user pick subset of columns to consider for duplicates (empty = consider all columns)
            cols_for_dup = st.multiselect("Consider these columns for duplicate detection (leave empty = all columns):",
                                        options=list(working_df.columns),
                                        default=[])

            # Preview duplicates (show up to 50)
            if total_dups > 0:
                if cols_for_dup:
                    dup_preview = working_df[working_df.duplicated(subset=cols_for_dup, keep=False)].head(50)
                else:
                    dup_preview = working_df[working_df.duplicated(keep=False)].head(50)
                st.markdown("**Preview of duplicate rows (showing up to 50)**")
                st.dataframe(dup_preview, use_container_width=True)
            else:
                st.info("No duplicate rows detected in the current dataset.")

            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Remove duplicates (all columns)" if not cols_for_dup else "Remove duplicates (selected columns)"):
                    before = working_df.shape[0]
                    # drop_duplicates returns the first occurrence by default and removes later ones;
                    # if user selected columns, pass subset; otherwise drop across all columns
                    if cols_for_dup:
                        cleaned = working_df.drop_duplicates(subset=cols_for_dup, keep="first")
                    else:
                        cleaned = working_df.drop_duplicates(keep="first")
                    after = cleaned.shape[0]
                    removed = before - after

                    # Update session-state and make the cleaned df the single source of truth
                    st.session_state["df"] = cleaned

                    # Optional: store a simple changelog (useful for UI)
                    st.session_state.setdefault("changelog", []).append(f"Removed {removed} duplicate rows (subset={cols_for_dup or 'ALL'})")

                    st.success(f"Removed {removed} duplicate rows. Dataset now has {after} rows.")

                    # Force Streamlit to rerun so Profiling and all tabs use the new df immediately
                    st.rerun()

            with col2:
                # Show recent changelog (if any)
                changelog = st.session_state.get("changelog", [])
                if changelog:
                    st.markdown("**Recent actions:**")
                    for item in changelog[-5:][::-1]:
                        st.write("•", item)
        # -----------------------------------------------------------------------------------


    # Encoding & Scaling
    elif selected == "Encoding & Scaling":
        st.subheader("Encoding & Scaling")
        num_cols, cat_cols = detect_numeric_cat(working_df)
        enc_choice = st.selectbox("Categorical encoding", options=["onehot", "label", "ordinal", "none"], index=0)
        scale_choice = st.selectbox("Scaling for numeric columns", options=["none", "standard", "minmax", "robust"], index=0)

        if st.button("Apply Encoding & Scaling"):
            df_curr = working_df.copy()
            if enc_choice != "none" and cat_cols:
                df_curr, enc_model = apply_encoding(df_curr, cat_cols, encoding=enc_choice)
                st.session_state["encoding"] = enc_model
            if scale_choice != "none":
                numeric_now, _ = detect_numeric_cat(df_curr)
                df_curr, scaler_model = apply_scaling(df_curr, numeric_now, scaler_name=scale_choice)
                st.session_state["scaler"] = scaler_model
            st.session_state["df"] = df_curr
            working_df = df_curr
            st.success("Encoding and scaling applied — preview below")
            st.dataframe(working_df.head(10), use_container_width=True)

    # Correlation
    elif selected == "Correlation":
        st.subheader("Correlation & Feature Optimization")
        num_cols, _ = detect_numeric_cat(working_df)
        if num_cols:
            fig = correlation_heatmap(working_df, num_cols)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            corr_thresh = st.slider("Drop features with absolute correlation above", 0.5, 0.99, 0.95)
            if st.button("Auto-drop correlated features"):
                corr_matrix = working_df[num_cols].corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper.columns if any(upper[column] > corr_thresh)]
                if to_drop:
                    before = working_df.shape[1]
                    working_df = working_df.drop(columns=to_drop)
                    st.session_state["df"] = working_df
                    after = working_df.shape[1]
                    st.success(f"Dropped {before - after} features due to high correlation: {to_drop}")
                else:
                    st.info("No features exceeded the correlation threshold.")
        else:
            st.info("No numeric columns available for correlation analysis")

    # Profiling
    elif selected == "Profiling":
        st.subheader("Automated EDA / Profiling")
        # Use working_df (most recent)
        if HAS_PROFILING:
            report = ProfileReport(working_df, explorative=True)
            st_profile_report(report)
        else:
            st.warning("ydata-profiling or streamlit-pandas-profiling not installed.")

    # Export
    elif selected == "Export":
        st.subheader("Export Cleaned Data")
        filename = st.text_input("Export filename", value="CleanserAI_cleaned.csv")
        if st.button("Prepare CSV for download"):
            # ensure df in session
            df_to_download = st.session_state.get("df", working_df)
            href = get_download_link(df_to_download, filename)
            st.markdown(f"⬇Download {filename}]({href})")
        if st.button("Save pipeline"):
            bundle = {
                "preprocessor": st.session_state.get("preprocessor", None),
                "encoding": st.session_state.get("encoding", None),
                "scaler": st.session_state.get("scaler", None)
            }
            with open("cleanserai_pipeline.pkl", "wb") as f:
                pickle.dump(bundle, f)
            st.success("Pipeline saved to cleanserai_pipeline.pkl")

else:
    st.info("Upload a dataset to begin.")
    st.markdown("- Try small CSVs like Titanic or Housing dataset.")

# Footer
st.markdown("""
<hr style='border:1px solid #eee;'>
<p style='text-align:center; color:gray; font-size:13px;'>
|CleanserAI © 2025|
</p>
""", unsafe_allow_html=True)
