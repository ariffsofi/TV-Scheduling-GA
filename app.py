from pathlib import Path
import hashlib
import inspect
import random

import pandas as pd
import streamlit as st
import ga_algorithm as ga

st.set_page_config(
    page_title="TV Scheduling | Dashboard",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {color-scheme:light}
.stApp, [data-testid="stAppViewContainer"] {
    background:#f6f8fb;
    color:#172b45;
}
[data-testid="stHeader"] {background:transparent}
.block-container {
    max-width:1200px;
    padding-top:2.1rem;
    padding-bottom:3rem;
}
[data-testid="stSidebar"] {
    background:#fff;
    border-right:1px solid #e8edf2;
}
[data-testid="stSidebar"] .block-container {
    padding-top:2rem;
}
h1,h2,h3 {
    color:#142b46!important;
    letter-spacing:-.025em;
}
h1 {
    font-size:2rem!important;
    font-weight:720!important;
}
h2 {
    font-size:1.28rem!important;
    font-weight:680!important;
}
h3 {
    font-size:1.03rem!important;
    font-weight:650!important;
}
p, label, [data-testid="stCaptionContainer"] {
    color:#53657a;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background:#fff;
    border-color:#e7ecf1!important;
    border-radius:13px!important;
}
[data-testid="stMetric"] {
    background:#fff;
    border:1px solid #e7ecf1;
    border-radius:12px;
    padding:1rem 1.15rem;
}
[data-testid="stMetricLabel"] {color:#65758a}
[data-testid="stMetricValue"] {
    color:#17324d;
    font-size:1.75rem;
}
.stButton button[kind="primary"] {
    background:#167e82!important;
    color:white!important;
    border:0!important;
    border-radius:8px!important;
    font-weight:600;
}
.stButton button[kind="primary"]:hover {
    background:#11696d!important;
}
.stButton button, .stDownloadButton button {
    border-radius:8px!important;
}
[data-testid="stFileUploader"] section {
    background:#fff;
    border-color:#d9e2eb;
    border-radius:10px;
}
[data-testid="stExpander"] {
    background:white;
    border:1px solid #e7ecf1;
    border-radius:10px;
}
[data-testid="stDataFrame"] {
    border-radius:8px;
    overflow:hidden;
}
[data-testid="stTabs"] button {font-weight:550}
[data-testid="stAlert"] {border-radius:9px}
.eyebrow {
    color:#148084;
    font-size:.75rem;
    font-weight:700;
    letter-spacing:.09em;
    text-transform:uppercase;
}
.page-lead {
    color:#6a7a8b;
    font-size:.94rem;
    margin-top:-.6rem;
    margin-bottom:1.4rem;
}
.section-lead {
    color:#718094;
    font-size:.88rem;
    margin-top:-.45rem;
    margin-bottom:1rem;
}
.brand {
    font-size:1.25rem;
    font-weight:750;
    color:#152d49;
    letter-spacing:-.04em;
    margin-bottom:.2rem;
}
.brand-sub {
    font-size:.77rem;
    color:#8290a1;
    margin-bottom:1.5rem;
}
.sidebar-note {
    font-size:.8rem;
    color:#7b8999;
}
.small-label {
    font-size:.76rem;
    letter-spacing:.07em;
    text-transform:uppercase;
    color:#8592a1;
    font-weight:650;
}
.footer-note {
    color:#94a0af;
    font-size:.78rem;
    margin-top:2rem;
}
@media(max-width:768px) {
    .block-container {
        padding:1.1rem .85rem 2rem;
    }
    h1 {font-size:1.65rem!important}
}
</style>
""", unsafe_allow_html=True)

HOURS = list(range(6, 24))
DEFAULT_TRIALS = [
    (0.80, 0.20),
    (0.70, 0.01),
    (0.95, 0.05),
]


# -----------------------------------------------------
# DATASET AND GENETIC ALGORITHM HELPERS
# -----------------------------------------------------

def validate_ratings(df):
    if df.shape[1] != 19:
        raise ValueError(
            "CSV needs one program-name column and "
            "18 hourly rating columns (06:00–23:00)."
        )

    names = df.iloc[:, 0].astype("string").str.strip()

    if (
        names.isna().any()
        or names.eq("").any()
        or names.duplicated().any()
    ):
        raise ValueError(
            "Program names must be non-empty and unique."
        )

    numeric = df.iloc[:, 1:].apply(
        pd.to_numeric,
        errors="raise",
    )

    if (
        numeric.isna().any().any()
        or not numeric.ge(0).all().all()
        or not numeric.le(1).all().all()
    ):
        raise ValueError(
            "All ratings must be numbers between 0 and 1, "
            "with no blanks."
        )

    return dict(
        zip(
            names.tolist(),
            numeric.astype(float).values.tolist(),
        )
    )


def set_ga_ratings(ratings):
    # Supports the earlier GA module and an updated version.
    if callable(getattr(ga, "update_ratings", None)):
        ga.update_ratings(ratings)
    else:
        ga.ratings = ratings
        ga.all_programs = list(ratings)
        ga.all_time_slots = HOURS.copy()


def timetable(schedule):
    if len(schedule) != len(HOURS):
        raise ValueError(
            f"Expected 18 scheduled hours, got {len(schedule)}."
        )

    return pd.DataFrame([
        {
            "Time": (
                f"{hour:02d}:00 – "
                f"{(hour + 1) % 24:02d}:00"
            ),
            "TV Program": program.replace("_", " ").title(),
            "Rating": round(
                float(ga.ratings[program][i]), 3
            ),
        }
        for i, (hour, program)
        in enumerate(zip(HOURS, schedule))
    ])


def execute_trials(parameters):
    results = []
    schedules = {}

    sig = inspect.signature(ga.genetic_algorithm)
    requires_initial = (
        "initial_schedule" in sig.parameters
    )

    programs = list(ga.ratings)

    if not programs:
        raise ValueError(
            "The ratings dataset has no programs."
        )

    for i, (co, mut) in enumerate(parameters, 1):
        random.seed(42)

        kwargs = {
            "generations": ga.GEN,
            "population_size": ga.POP,
            "crossover_rate": co,
            "mutation_rate": mut,
            "elitism_size": ga.EL_S,
        }

        if requires_initial:
            initial = [
                programs[j % len(programs)]
                for j in range(len(HOURS))
            ]
            random.shuffle(initial)

            candidate = ga.genetic_algorithm(
                initial,
                **kwargs,
            )
        else:
            candidate = ga.genetic_algorithm(
                **kwargs
            )

        if len(candidate) != len(HOURS):
            raise ValueError(
                "GA returned a schedule of the wrong length. "
                "Check ga_algorithm.py."
            )

        name = f"Experiment {i}"
        score = float(ga.fitness_function(candidate))

        results.append({
            "Experiment": name,
            "Crossover Rate": co,
            "Mutation Rate": mut,
            "Total Rating": round(score, 3),
        })

        schedules[name] = timetable(candidate)

    return pd.DataFrame(results), schedules


# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="brand">TV Scheduling</div>'
        '<div class="brand-sub">Genetic Algorithm Optimizer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="small-label">WORKSPACE</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Experiment setup",
            "Results & schedules",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        '<div class="small-label">ALGORITHM SETTINGS</div>',
        unsafe_allow_html=True,
    )

    st.caption(f"Generations: {ga.GEN}")
    st.caption(f"Population size: {ga.POP}")
    st.caption(f"Elitism size: {ga.EL_S}")

    st.divider()

    st.markdown(
        '<div class="sidebar-note">'
        'TV Scheduling Optimization<br>'
        'Computational Intelligence'
        '</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------
# LOAD DATASET
# -----------------------------------------------------

dataset_path = Path(__file__).resolve().parent / (
    "program_ratings_modified.csv"
)

if "uploaded_csv" not in st.session_state:
    st.session_state["uploaded_csv"] = None

try:
    if st.session_state["uploaded_csv"] is not None:
        csv_bytes = st.session_state["uploaded_csv"]
        from io import BytesIO
        ratings_df = pd.read_csv(BytesIO(csv_bytes))
        dataset_name = "Uploaded ratings CSV"
    else:
        ratings_df = pd.read_csv(dataset_path)
        dataset_name = "Default modified ratings"

    active_ratings = validate_ratings(ratings_df)
    set_ga_ratings(active_ratings)

except Exception as error:
    st.error(f"Unable to load ratings: {error}")
    st.stop()

dataset_bytes = ratings_df.to_csv(
    index=False
).encode("utf-8")

dataset_id = hashlib.sha256(
    dataset_bytes
).hexdigest()


# -----------------------------------------------------
# OVERVIEW PAGE
# -----------------------------------------------------

if page == "Overview":
    st.markdown(
        '<div class="eyebrow">Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.title("TV Scheduling Overview")

    st.markdown(
        '<div class="page-lead">'
        'Manage program ratings and optimize your '
        'television schedule using a genetic algorithm.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "TV Programs",
        len(ga.all_programs),
    )

    col2.metric(
        "Time Slots",
        len(ga.all_time_slots),
    )

    col3.metric(
        "Experiments",
        3,
    )

    st.write("")
    st.subheader("Program ratings dataset")

    st.markdown(
        '<div class="section-lead">'
        'Upload a modified CSV file or use the '
        'existing program ratings.'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(f"**Active dataset:** {dataset_name}")

        uploaded_file = st.file_uploader(
            "Upload ratings CSV",
            type=["csv"],
            help=(
                "CSV must contain one program column "
                "and 18 hourly rating columns."
            ),
        )

        if uploaded_file is not None:
            new_bytes = uploaded_file.getvalue()

            if (
                st.session_state["uploaded_csv"]
                != new_bytes
            ):
                st.session_state["uploaded_csv"] = new_bytes
                st.rerun()

        if st.session_state["uploaded_csv"] is not None:
            if st.button("Use default dataset"):
                st.session_state["uploaded_csv"] = None
                st.rerun()

    with st.expander("View program ratings"):
        st.dataframe(
            ratings_df,
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download ratings CSV",
            data=dataset_bytes,
            file_name="active_program_ratings.csv",
            mime="text/csv",
        )

    st.write("")
    st.subheader("Getting started")

    with st.container(border=True):
        st.markdown(
            "**1. Review the ratings dataset**"
        )
        st.caption(
            "Check the program ratings used by the algorithm."
        )

        st.markdown(
            "**2. Configure three experiments**"
        )
        st.caption(
            "Choose crossover and mutation rates."
        )

        st.markdown(
            "**3. Run and compare schedules**"
        )
        st.caption(
            "Evaluate fitness scores and generated timetables."
        )

        if st.button(
            "Configure experiments",
            type="primary",
        ):
            st.session_state["go_setup"] = True
            st.rerun()


# -----------------------------------------------------
# EXPERIMENT SETUP PAGE
# -----------------------------------------------------

elif page == "Experiment setup":
    st.markdown(
        '<div class="eyebrow">Optimization</div>',
        unsafe_allow_html=True,
    )

    st.title("Experiment Setup")

    st.markdown(
        '<div class="page-lead">'
        'Set crossover and mutation rates for '
        'three genetic algorithm trials.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Algorithm parameters")

    st.markdown(
        '<div class="section-lead">'
        'Each experiment uses the same generations, '
        'population size, and elitism settings.'
        '</div>',
        unsafe_allow_html=True,
    )

    for i, (default_co, default_mut) in enumerate(
        DEFAULT_TRIALS
    ):
        with st.container(border=True):
            st.markdown(
                f"### Experiment {i + 1}"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.number_input(
                    "Crossover Rate (CO_R)",
                    min_value=0.00,
                    max_value=0.95,
                    value=default_co,
                    step=0.05,
                    format="%.2f",
                    key=f"co_{i}",
                )

            with col2:
                st.number_input(
                    "Mutation Rate (MUT_R)",
                    min_value=0.01,
                    max_value=0.20,
                    value=default_mut,
                    step=0.01,
                    format="%.2f",
                    key=f"mut_{i}",
                )

    st.caption(
        "Default mutation rate: 0.20. "
        "Additional trial values include 0.01–0.05."
    )

    st.write("")

    if st.button(
        "Run three experiments",
        type="primary",
        use_container_width=True,
    ):
        params = [
            (
                st.session_state[f"co_{i}"],
                st.session_state[f"mut_{i}"],
            )
            for i in range(3)
        ]

        with st.spinner(
            "Running genetic algorithm experiments..."
        ):
            try:
                results, schedules = execute_trials(params)

                st.session_state["ga_results"] = results
                st.session_state["ga_schedules"] = schedules
                st.session_state["ga_run_params"] = params
                st.session_state["ga_dataset_id"] = dataset_id

                st.success(
                    "All three experiments completed successfully."
                )

                st.info(
                    "Open Results & schedules from the sidebar "
                    "to view the comparison and timetables."
                )

            except Exception as error:
                st.error(
                    f"Optimization failed: {error}"
                )


# -----------------------------------------------------
# RESULTS AND SCHEDULES PAGE
# -----------------------------------------------------

elif page == "Results & schedules":
    st.markdown(
        '<div class="eyebrow">Analysis</div>',
        unsafe_allow_html=True,
    )

    st.title("Results & Schedules")

    st.markdown(
        '<div class="page-lead">'
        'Compare experiment fitness scores and '
        'review the generated TV timetables.'
        '</div>',
        unsafe_allow_html=True,
    )

    if "ga_results" not in st.session_state:
        st.info(
            "No experiment results yet. "
            "Go to Experiment setup and run the three trials."
        )

    else:
        current_params = [
            (
                st.session_state.get(
                    f"co_{i}",
                    DEFAULT_TRIALS[i][0],
                ),
                st.session_state.get(
                    f"mut_{i}",
                    DEFAULT_TRIALS[i][1],
                ),
            )
            for i in range(3)
        ]

        is_current = (
            st.session_state.get("ga_dataset_id")
            == dataset_id
            and st.session_state.get("ga_run_params")
            == current_params
        )

        if not is_current:
            st.warning(
                "The ratings or parameters have changed "
                "since these results were generated. "
                "Run the trials again to refresh them."
            )

        else:
            results_df = st.session_state["ga_results"]
            schedules = st.session_state["ga_schedules"]

            best_score = results_df["Total Rating"].max()

            leaders = results_df.loc[
                results_df["Total Rating"] == best_score,
                "Experiment",
            ].tolist()

            st.subheader("Experiment performance")

            col1, col2, col3 = st.columns(3)

            for col, (_, record) in zip(
                (col1, col2, col3),
                results_df.iterrows(),
            ):
                col.metric(
                    record["Experiment"],
                    f'{record["Total Rating"]:.2f}',
                )

            st.write("")

            with st.container(border=True):
                st.markdown("**Highest fitness**")

                st.markdown(
                    f"**{', '.join(leaders)}** · "
                    f"Total rating **{best_score:.2f}**"
                )

                st.caption(
                    "A higher total rating indicates "
                    "a better schedule under the selected dataset."
                )

            st.subheader("Fitness comparison")

            st.bar_chart(
                results_df.set_index("Experiment")[
                    "Total Rating"
                ],
                color="#17878b",
            )

            st.dataframe(
                results_df,
                hide_index=True,
                use_container_width=True,
            )

            st.download_button(
                "Download comparison CSV",
                data=results_df.to_csv(index=False),
                file_name="experiment_comparison.csv",
                mime="text/csv",
            )

            st.divider()

            st.subheader("TV programming schedules")

            st.markdown(
                '<div class="section-lead">'
                'Each table lists the scheduled programs '
                'from 06:00 through midnight.'
                '</div>',
                unsafe_allow_html=True,
            )

            tabs = st.tabs([
                "Experiment 1",
                "Experiment 2",
                "Experiment 3",
            ])

            for i, tab in enumerate(tabs):
                with tab:
                    name = f"Experiment {i + 1}"

                    co, mut = st.session_state[
                        "ga_run_params"
                    ][i]

                    st.caption(
                        f"Crossover {co:.2f} · "
                        f"Mutation {mut:.2f}"
                    )

                    schedule_df = schedules[name]

                    st.dataframe(
                        schedule_df,
                        hide_index=True,
                        use_container_width=True,
                        height=490,
                    )

                    st.download_button(
                        "Download schedule CSV",
                        data=schedule_df.to_csv(
                            index=False
                        ),
                        file_name=(
                            f"experiment_{i + 1}_schedule.csv"
                        ),
                        mime="text/csv",
                        key=f"download_{i}",
                    )


st.markdown(
    '<div class="footer-note">'
    'TV Scheduling Optimization · Genetic Algorithm'
    '</div>',
    unsafe_allow_html=True,
)
