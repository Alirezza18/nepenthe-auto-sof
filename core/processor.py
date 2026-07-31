import pandas as pd
import numpy as np
import chardet
import io

def analyze_manifold(df):
    """Scientific diagnostics for simulation-grade data matrices."""
    numeric_df = df.select_dtypes(include=[np.number])
    return {
        "observations": len(df),
        "dimensions": len(df.columns),
        "sparsity": f"{(df.isna().sum().sum() / (df.size if df.size > 0 else 1) * 100):.2f}%",
        "numeric_features": numeric_df.columns.tolist(),
        "constant_factors": [c for c in numeric_df.columns if df[c].nunique() <= 1],
        "null_entries": int(df.isna().sum().sum())
    }

def process_ingested_file(uploaded_file):
    """Smart ingestion engine with encoding recovery."""
    try:
        content = uploaded_file.read()
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            detection = chardet.detect(content[:50000])
            encoding = detection['encoding'] or 'utf-8'
            df = pd.read_csv(io.BytesIO(content), encoding=encoding, encoding_errors='replace')
        
        df.columns = df.columns.astype(str).str.replace("\xa0"," ").str.strip()
        return df
    except Exception as e:
        return str(e)