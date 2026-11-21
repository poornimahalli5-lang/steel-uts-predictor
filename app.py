import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Steel UTS Predictor", layout="wide")
st.title("Steel Ultimate Tensile Strength (UTS) Predictor")

FILE_PATH = "Steel database with labelled clusters (1).xlsx"

@st.cache_data
def load_data():
    xls = pd.ExcelFile(FILE_PATH)
    df = pd.read_excel(FILE_PATH, sheet_name=xls.sheet_names[0]).dropna(axis=1, how="all").copy()
    df["Name"] = df["Name"].astype(str).str.strip()
    return df

df = load_data()
chem_features = ["Al", "Cu", "Mn", "N", "Ni", "Ti", "S", "Fe", "Zr", "P", "Si", "V", "Mo", "Co", "C", "Nb", "B", "Cr"]
cat_features = ["Cluster label"]
target_col = "(Ultimate) Tensile strength (MPa)"

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), chem_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
])

tab1, tab2 = st.tabs(["1. Unseen Grade Holdout Check", "2. Custom Specimen Predictor"])

with tab1:
    grades = sorted(df["Name"].unique())
    selected_grade = st.selectbox("Select Steel Grade (100% excluded from training):", grades, index=grades.index("AISI 1060 Steel"))
    only_as_received = st.checkbox("Show only As-Received (Hot Rolled / Cold Drawn / As Rolled) rows")

    if st.button("Run Unseen Grade Prediction"):
        digits = "".join(filter(str.isdigit, selected_grade))
        train_df = df[~df["Name"].str.contains(digits, na=False)].copy() if digits else df[df["Name"] != selected_grade].copy()
        holdout_df = df[df["Name"] == selected_grade].copy()

        pipe = Pipeline([("prep", preprocessor), ("model", RandomForestRegressor(n_estimators=200, random_state=42))])
        pipe.fit(train_df[chem_features + cat_features], train_df[target_col])
        preds = pipe.predict(holdout_df[chem_features + cat_features])

        res = holdout_df[["Name", "Processing condition", "C", target_col]].copy()
        res.rename(columns={target_col: "Actual_UTS_MPa"}, inplace=True)
        res["Predicted_UTS_MPa"] = np.round(preds, 1)
        res["Abs_Error_MPa"] = np.round(np.abs(res["Actual_UTS_MPa"] - res["Predicted_UTS_MPa"]), 1)
        res["Pct_Error_%"] = np.round((res["Abs_Error_MPa"] / res["Actual_UTS_MPa"]) * 100, 1)

        if only_as_received:
            res = res[res["Processing condition"].str.contains("Hot Rolled|Cold Drawn|As Rolled", case=False, na=False)]

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Actual UTS", f"{res['Actual_UTS_MPa'].mean():.1f} MPa")
        c2.metric("Mean Predicted UTS", f"{res['Predicted_UTS_MPa'].mean():.1f} MPa")
        c3.metric("Mean Absolute Error", f"{res['Abs_Error_MPa'].mean():.1f} MPa ({res['Pct_Error_%'].mean():.1f}%)")

        st.dataframe(res, use_container_width=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(res["Actual_UTS_MPa"], res["Predicted_UTS_MPa"], color="darkred", s=65)
        min_v = min(res["Actual_UTS_MPa"].min(), res["Predicted_UTS_MPa"].min()) - 40
        max_v = max(res["Actual_UTS_MPa"].max(), res["Predicted_UTS_MPa"].max()) + 40
        ax.plot([min_v, max_v], [min_v, max_v], "k--", label="Ideal 1:1")
        ax.set_xlabel("Actual UTS (MPa)")
        ax.set_ylabel("Predicted UTS (MPa)")
        ax.set_title(f"{selected_grade}: Predicted vs. Actual UTS")
        ax.legend()
        ax.grid(True, alpha=0.4)
        st.pyplot(fig)

with tab2:
    col1, col2, col3 = st.columns(3)
    c_val = col1.number_input("Carbon C (wt.%)", 0.0, 2.0, 0.60, 0.01)
    mn_val = col2.number_input("Manganese Mn (wt.%)", 0.0, 3.0, 0.75, 0.05)
    si_val = col3.number_input("Silicon Si (wt.%)", 0.0, 3.0, 0.22, 0.05)
    cr_val = col1.number_input("Chromium Cr (wt.%)", 0.0, 5.0, 0.0, 0.05)
    ni_val = col2.number_input("Nickel Ni (wt.%)", 0.0, 5.0, 0.0, 0.05)
    mo_val = col3.number_input("Molybdenum Mo (wt.%)", 0.0, 2.0, 0.0, 0.05)
    cluster = st.selectbox("Processing Condition (Cluster Label)", sorted(df["Cluster label"].unique()))

    if st.button("Predict New Specimen UTS"):
        full_pipe = Pipeline([("prep", preprocessor), ("model", RandomForestRegressor(n_estimators=200, random_state=42))])
        full_pipe.fit(df[chem_features + cat_features], df[target_col])
        fe_val = max(0.0, 100.0 - (c_val + mn_val + si_val + cr_val + ni_val + mo_val + 0.06))
        row = pd.DataFrame([{
            "Al": 0.0, "Cu": 0.0, "Mn": mn_val, "N": 0.0, "Ni": ni_val, "Ti": 0.0,
            "S": 0.03, "Fe": fe_val, "Zr": 0.0, "P": 0.03, "Si": si_val, "V": 0.0,
            "Mo": mo_val, "Co": 0.0, "C": c_val, "Nb": 0.0, "B": 0.0, "Cr": cr_val,
            "Cluster label": cluster
        }])
        pred = full_pipe.predict(row)[0]
        st.success(f"Predicted Ultimate Tensile Strength (UTS): **{pred:.1f} MPa**")
        
