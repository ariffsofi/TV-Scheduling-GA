
import io
import random
import hashlib

import pandas as pd
import streamlit as st

import ga_algorithm as ga


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TV Scheduling | GA Optimizer",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# MINIMALIST LIGHT THEME
# ============================================================

st.markdown("""
<style>
    :root {
        color-scheme: light;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background-color: #F7F9FA;
        color: #1E293B;
    }

    [data-testid="stHeader"] {
        background-color: #F7F9FA;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #172B3A !important;
        letter-spacing: -0.025em;
    }

    h1 {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    h2, h3 {
        font-weight: 650 !important;
    }

    p, label, .stMarkdown {
        color: #334155;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B;
    }

    div[data-testid="stMetricValue"] {
        color: #172B3A;
    }

    div[data-testid="stFileUploader"] {
        background: #FFFFFF;
        border-radius: 10px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
    }

    div[data-testid="stNumberInput"] input {
        background: #FFFFFF;
        color: #172B3A;
        border-radius: 8px;
    }

    div.stButton > button[kind="primary"] {
        background-color: #147D80;
        color: #FFFFFF;
        border: none;
        border-radius: 9px;
        padding: 0.65rem 1.4rem;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #10696B;
        color: #FFFFFF;
    }

    div.stDownloadButton > button {
        background-color: #FFFFFF;
        color: #147D80;
        border: 1px solid #B8D9D8;
        border-radius: 8px;
    }

    div.stDownloadButton > button:hover {
        color: #10696B;
        border-color: #147D80;
    }

    .app-subtitle {
        color: #64748B;
        font-size: 0.95rem;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }

    .section-label {
        color: #147D80;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }

    .footer-text {
        color: #94A3B8;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 2rem;
    }

    hr {
        border-color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def prepare_ratings(df):
    if df.shape[1] != 19:
        raise ValueError(
            "CSV must contain one program column "
            "and 18 hourly rating columns."
        )

    if df.empty:
        raise ValueError("The CSV contains no programs.")

    if df.iloc[:, 0].isna().any():
        raise ValueError("Program names cannot be empty.")

    programs = df.iloc[:, 0].astype(str).str.strip()

    if programs.eq("").any() or programs.duplicated().any():
        raise ValueError(
            "Program names must be unique and non-empty."
        )

    numeric = df.iloc[:, 1:].apply(
        pd.to_numeric,
        errors="raise"
    )

    if numeric.isna().any().any():
        raise ValueError("Ratings cannot contain empty values.")

    if not numeric.ge(0).all().all() or not numeric.le(1).all().all():
        raise ValueError(
            "All ratings must be between 0 and 1."
        )

    return {
        program: [float(value) for value in row]
        for program, row in zip(
            programs,
            numeric.to_numpy().tolist()
        )
    }


def schedule_to_dataframe(schedule):
    rows = []

    for index, program in enumerate(schedule):
        start_hour = ga.all_time_slots[index]
        end_hour = (start_hour + 1) % 24

        rows.append({
            "Time": (
                f"{start_hour:02d}:00 - "
                f"{end_hour:02d}:00"
            ),
            "TV Program": program.replace("_", " ").title(),
            "Rating": round(
                ga.ratings[program][index], 3
            )
        })

    return pd.DataFrame(rows)


def calculate_dataset_id(data):
    return hashlib.sha256(data).hexdigest()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="section-label">GENETIC ALGORITHM OPTIMIZER</div>',
    unsafe_allow_html=True
)

st.title("📺 TV Scheduling")

st.markdown(
    '<div class="app-subtitle">'
    'Optimize television programming and compare '
    'Genetic Algorithm experiments.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DATASET
# ============================================================

st.subheader("Program Ratings")

uploaded_file = st.file_uploader(
    "Upload a program ratings CSV",
    type=["csv"],
    help="The CSV must contain 18 hourly ratings per program."
)

try:
    if uploaded_file is not None:
        dataset_bytes = uploaded_file.getvalue()
        ratings_df = pd.read_csv(io.BytesIO(dataset_bytes))
        dataset_name = uploaded_file.name
    else:
        with open("program_ratings_modified.csv", "rb") as file:
            dataset_bytes = file.read()

        ratings_df = pd.read_csv(io.BytesIO(dataset_bytes))
        dataset_name = "program_ratings_modified.csv"

    new_ratings = prepare_ratings(ratings_df)
    ga.update_ratings(new_ratings)

except Exception as error:
    st.error(f"Dataset error: {error}")
    st.stop()

dataset_id = calculate_dataset_id(dataset_bytes)

st.success(f"Dataset ready: {dataset_name}")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("TV Programs", len(new_ratings))

with metric2:
    st.metric("Time Slots", len(ga.all_time_slots))

with metric3:
    st.metric("Experiments", 3)

with st.expander("Preview program ratings"):
    st.dataframe(
        ratings_df,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================

st.divider()

st.subheader("Experiment Parameters")

st.caption(
    "Customize the crossover and mutation rates "
    "for each experiment."
)

default_rates = [
    (0.80, 0.20),
    (0.70, 0.01),
    (0.95, 0.05)
]

experiment_parameters = []

for i, (default_co, default_mut) in enumerate(default_rates):
    with st.container(border=True):
        st.markdown(f"**Experiment {i + 1:02d}**")

        col1, col2 = st.columns(2)

        with col1:
            co_rate = st.number_input(
                "Crossover Rate (CO_R)",
                min_value=0.00,
                max_value=0.95,
                value=default_co,
                step=0.05,
                format="%.2f",
                key=f"co_rate_{i}"
            )

        with col2:
            mut_rate = st.number_input(
                "Mutation Rate (MUT_R)",
                min_value=0.01,
                max_value=0.20,
                value=default_mut,
                step=0.01,
                format="%.2f",
                key=f"mut_rate_{i}"
            )

        experiment_parameters.append(
            (co_rate, mut_rate)
        )

st.caption(
    "The mutation default of 0.20 is retained. "
    "The 0.01–0.05 range is also available for trials."
)


# ============================================================
# RUN GENETIC ALGORITHM
# ============================================================

st.divider()

st.subheader("Run Optimization")

st.write(
    "Run three experiments and compare the generated "
    "TV schedules using their total ratings."
)

run_clicked = st.button(
    "▶ Run Three Experiments",
    type="primary",
    use_container_width=True
)

if run_clicked:
    results = []
    schedules = {}

    with st.spinner("Optimizing TV schedules..."):
        for i, (co_rate, mut_rate) in enumerate(
            experiment_parameters
        ):
            # Same seed for reproducible comparisons
            random.seed(42)

            schedule = ga.genetic_algorithm(
                generations=ga.GEN,
                population_size=ga.POP,
                crossover_rate=co_rate,
                mutation_rate=mut_rate,
                elitism_size=ga.EL_S
            )

            total_rating = ga.fitness_function(schedule)
            name = f"Experiment {i + 1}"

            results.append({
                "Experiment": name,
                "Crossover Rate": co_rate,
                "Mutation Rate": mut_rate,
                "Total Rating": round(total_rating, 3)
            })

            schedules[name] = schedule_to_dataframe(
                schedule
            )

    st.session_state["ga_results"] = results
    st.session_state["ga_schedules"] = schedules
    st.session_state["ga_dataset_id"] = dataset_id
    st.session_state["ga_parameters"] = (
        experiment_parameters.copy()
    )


# ============================================================
# EXPERIMENT RESULTS
# ============================================================

if "ga_results" in st.session_state:
    inputs_match = (
        st.session_state["ga_dataset_id"] == dataset_id
        and st.session_state["ga_parameters"]
        == experiment_parameters
    )

    if not inputs_match:
        st.warning(
            "Your dataset or parameters have changed. "
            "Run the experiments again to refresh the results."
        )

    else:
        st.divider()

        st.subheader("Experiment Comparison")

        results_df = pd.DataFrame(
            st.session_state["ga_results"]
        )

        best_rating = results_df["Total Rating"].max()

        best_experiments = results_df.loc[
            results_df["Total Rating"] == best_rating,
            "Experiment"
        ].tolist()

        best_label = ", ".join(best_experiments)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Highest Total Rating",
                f"{best_rating:.3f}"
            )

        with col2:
            st.metric(
                "Best Experiment",
                best_label
            )

        st.dataframe(
            results_df,
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "Higher total ratings indicate better "
            "performance under the selected fitness function."
        )

        st.download_button(
            "Download Experiment Comparison",
            data=results_df.to_csv(index=False),
            file_name="experiment_comparison.csv",
            mime="text/csv"
        )

        # ====================================================
        # TV SCHEDULES
        # ====================================================

        st.divider()

        st.subheader("TV Programming Schedules")

        st.caption(
            "Generated schedules from 06:00 to 00:00."
        )

        tabs = st.tabs([
            "Experiment 01",
            "Experiment 02",
            "Experiment 03"
        ])

        for i, tab in enumerate(tabs):
            name = f"Experiment {i + 1}"

            schedule_df = (
                st.session_state["ga_schedules"][name]
            )

            with tab:
                st.markdown(f"**{name} — TV Timetable**")

                co_rate, mut_rate = experiment_parameters[i]

                st.caption(
                    f"Crossover: {co_rate:.2f}  |  "
                    f"Mutation: {mut_rate:.2f}"
                )

                st.dataframe(
                    schedule_df,
                    hide_index=True,
                    use_container_width=True,
                    height=560
                )

                st.download_button(
                    "Download Schedule CSV",
                    data=schedule_df.to_csv(index=False),
                    file_name=(
                        f"experiment_{i + 1}_schedule.csv"
                    ),
                    mime="text/csv",
                    key=f"download_schedule_{i}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer-text">'
    'TV Scheduling Optimization · Genetic Algorithm'
    '</div>',
    unsafe_allow_html=True
)
