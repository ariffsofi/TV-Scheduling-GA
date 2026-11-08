
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="TV Scheduling - Genetic Algorithm",
    page_icon="📺",
    layout="wide"
)

st.title("📺 TV Scheduling Using Genetic Algorithm")

st.write(
    "Generate optimized television schedules "
    "using Genetic Algorithm."
)

st.subheader("1. Genetic Algorithm Parameters")

col1, col2 = st.columns(2)

with col1:
    co_rate = st.number_input(
        "Crossover Rate (CO_R)",
        min_value=0.0,
        max_value=0.95,
        value=0.80,
        step=0.05,
        format="%.2f"
    )

with col2:
    mut_rate = st.number_input(
        "Mutation Rate (MUT_R)",
        min_value=0.01,
        max_value=0.20,
        value=0.20,
        step=0.01,
        format="%.2f"
    )

st.caption(
    "Mutation default: 0.20. The assignment also "
    "mentions an adjustable range of 0.01 to 0.05."
)

st.subheader("2. Program Ratings Dataset")

uploaded_file = st.file_uploader(
    "Upload your modified program ratings CSV",
    type=["csv"]
)

if uploaded_file is not None:
    ratings_df = pd.read_csv(uploaded_file)

    st.success("CSV file uploaded successfully!")
    st.dataframe(
        ratings_df,
        hide_index=True,
        use_container_width=True
    )
else:
    st.info("Please upload a CSV file to continue.")
