
import streamlit as st
import pandas as pd
import random
import hashlib

import ga_algorithm as ga


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="TV Scheduling | GA Optimizer",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================================
# MINIMALIST LIGHT THEME
# =====================================================

st.markdown("""
<style>
:root {
    color-scheme: light;
}

.stApp {
    background-color: #F7F9FC;
    color: #172B4D;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    color: #172B4D !important;
    letter-spacing: -0.025em;
}

h1 {
    font-size: clamp(1.8rem, 4vw, 2.6rem) !important;
    font-weight: 750 !important;
}

h2, h3 {
    font-weight: 700 !important;
}

p {
    line-height: 1.65;
}

[data-testid="stHeader"] {
    background: #F7F9FC;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-color: #E2E8F0 !important;
    border-radius: 14px !important;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    padding: 1.1rem;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
}

[data-testid="stMetricValue"] {
    color: #172B4D !important;
    font-weight: 700;
}

.stButton > button[kind="primary"] {
    background: #167A80 !important;
    color: white !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 0.65rem 1.5rem !important;
    font-weight: 650 !important;
}

.stButton > button[kind="primary"]:hover {
    background: #11656A !important;
}

.stDownloadButton > button {
    border: 1px solid #CBD5E1 !important;
    border-radius: 9px !important;
    color: #172B4D !important;
    background: white !important;
}

[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border-radius: 12px;
}

[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

[data-testid="stNumberInput"] input {
    border-radius: 8px;
}

[data-testid="stExpander"] {
    background: #FFFFFF;
    border-radius: 10px;
}

.section-caption {
    color: #64748B;
    font-size: 0.92rem;
    margin-top: -0.5rem;
    margin-bottom: 1.2rem;
}

.hero-tag {
    display: inline-block;
    background: #E5F3F2;
    color: #167A80;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}

.footer {
    color: #94A3B8;
    font-size: 0.8rem;
    text-align: center;
    padding-top: 1rem;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1.2rem 1rem 3rem;
    }

    h1 {
        font-size: 1.8rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def prepare_ratings(df):
    """Validate CSV and convert it into GA format."""

    if df.shape[1] != 19:
        raise ValueError(
            "CSV must contain one program column "
            "and 18 hourly rating columns."
        )

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
    """Convert a GA schedule into a TV timetable."""

    rows = []

    for index, program in enumerate(schedule):
        start_hour = ga.all_time_slots[index]
        end_hour = (start_hour + 1) % 24

        rows.append({
            "Time": (
                f"{start_hour:02d}:00 - "
                f"{end_hour:02d}:00"
            ),
            "TV Program": (
                program.replace("_", " ").title()
            ),
            "Rating": round(
                ga.ratings[program][index], 3
            )
        })

    return pd.DataFrame(rows)


def run_experiments(parameters):
    """Run three GA trials and return their results."""

    results = []
    schedules = {}

    for index, (co_rate, mut_rate) in enumerate(parameters):
        # Same seed makes comparisons reproducible.
        random.seed(42)

        schedule = ga.genetic_algorithm(
            generations=ga.GEN,
            population_size=ga.POP,
            crossover_rate=co_rate,
            mutation_rate=mut_rate,
            elitism_size=ga.EL_S
        )

        total_rating = ga.fitness_function(schedule)
        name = f"Experiment {index + 1}"

        results.append({
            "Experiment": name,
            "Crossover Rate": co_rate,
            "Mutation Rate": mut_rate,
            "Total Rating": round(total_rating, 3)
        })

        schedules[name] = schedule_to_dataframe(schedule)

    return results, schedules


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="hero-tag">GENETIC ALGORITHM OPTIMIZER</div>',
    unsafe_allow_html=True
)

st.title("📺 TV Scheduling Dashboard")

st.markdown(
    """
    <div class="section-caption">
        Optimize television program scheduling,
        configure genetic algorithm parameters,
        and compare experiment performance.
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# DATASET
# =====================================================

st.subheader("01 / Program Ratings Dataset")

st.markdown(
    '<div class="section-caption">'
    'Upload a modified ratings CSV or use the default dataset.'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload program ratings CSV",
    type=["csv"],
    help="The file must contain program names and 18 hourly ratings."
)

try:
    if uploaded_file is not None:
        ratings_df = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name
    else:
        ratings_df = pd.read_csv(
            "program_ratings_modified.csv"
        )
        dataset_name = "Default Modified Ratings"

    new_ratings = prepare_ratings(ratings_df)
    ga.update_ratings(new_ratings)

except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()


# Dataset identifier for detecting changes
dataset_bytes = ratings_df.to_csv(
    index=False
).encode("utf-8")

dataset_id = hashlib.sha256(
    dataset_bytes
).hexdigest()


# =====================================================
# OVERVIEW METRICS
# =====================================================

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "TV Programs",
        len(ga.all_programs)
    )

with metric2:
    st.metric(
        "Hourly Time Slots",
        len(ga.all_time_slots)
    )

with metric3:
    st.metric(
        "Configured Experiments",
        3
    )

st.success(f"Active dataset: {dataset_name}")

with st.expander("View Program Ratings Dataset"):
    st.dataframe(
        ratings_df,
        hide_index=True,
        use_container_width=True
    )

    st.download_button(
        "Download Active Ratings CSV",
        data=dataset_bytes,
        file_name="active_program_ratings.csv",
        mime="text/csv"
    )


# =====================================================
# EXPERIMENT CONFIGURATION
# =====================================================

st.divider()

st.subheader("02 / Experiment Configuration")

st.markdown(
    '<div class="section-caption">'
    'Adjust crossover and mutation rates for each experiment.'
    '</div>',
    unsafe_allow_html=True
)

default_parameters = [
    (0.80, 0.20),
    (0.70, 0.01),
    (0.95, 0.05)
]

experiment_parameters = []

for index, (default_co, default_mut) in enumerate(
    default_parameters
):
    with st.container(border=True):
        st.markdown(
            f"#### Experiment {index + 1}"
        )

        col1, col2 = st.columns(2)

        with col1:
            co_rate = st.number_input(
                "Crossover Rate (CO_R)",
                min_value=0.00,
                max_value=0.95,
                value=default_co,
                step=0.05,
                format="%.2f",
                key=f"co_rate_{index}"
            )

        with col2:
            mut_rate = st.number_input(
                "Mutation Rate (MUT_R)",
                min_value=0.01,
                max_value=0.20,
                value=default_mut,
                step=0.01,
                format="%.2f",
                key=f"mut_rate_{index}"
            )

        experiment_parameters.append(
            (co_rate, mut_rate)
        )

st.caption(
    "The default mutation rate is 0.20. "
    "The assignment also specifies trial values "
    "between 0.01 and 0.05."
)


# =====================================================
# RUN ALGORITHM
# =====================================================

st.divider()

st.subheader("03 / Run Optimization")

st.markdown(
    '<div class="section-caption">'
    'Execute three genetic algorithm trials '
    'using the configured parameters.'
    '</div>',
    unsafe_allow_html=True
)

if st.button(
    "▶ Run Three Experiments",
    type="primary",
    use_container_width=True
):
    with st.spinner(
        "Optimizing TV schedules..."
    ):
        try:
            results, schedules = run_experiments(
                experiment_parameters
            )

            st.session_state["ga_results"] = results
            st.session_state["ga_schedules"] = schedules
            st.session_state["ga_dataset_id"] = dataset_id
            st.session_state["ga_parameters"] = (
                experiment_parameters.copy()
            )

        except Exception as error:
            st.error(
                f"Optimization failed: {error}"
            )
            st.stop()


# =====================================================
# RESULTS
# =====================================================

if "ga_results" in st.session_state:

    results_are_current = (
        st.session_state.get("ga_dataset_id") == dataset_id
        and st.session_state.get("ga_parameters")
        == experiment_parameters
    )

    if not results_are_current:
        st.warning(
            "The dataset or parameters have changed. "
            "Run the experiments again to refresh the results."
        )

    else:
        results_df = pd.DataFrame(
            st.session_state["ga_results"]
        )

        schedules = st.session_state["ga_schedules"]

        st.divider()

        st.subheader("04 / Experiment Results")

        st.markdown(
            '<div class="section-caption">'
            'Compare the fitness scores of all three trials.'
            '</div>',
            unsafe_allow_html=True
        )

        best_index = results_df[
            "Total Rating"
        ].idxmax()

        best_result = results_df.loc[best_index]

        score1, score2, score3 = st.columns(3)

        for index, column in enumerate(
            [score1, score2, score3]
        ):
            with column:
                st.metric(
                    f"Experiment {index + 1}",
                    f"{results_df.iloc[index]['Total Rating']:.2f}"
                )

        with st.container(border=True):
            st.markdown("#### Best Performing Experiment")

            st.markdown(
                f"**{best_result['Experiment']}** "
                f"achieved the highest total rating "
                f"of **{best_result['Total Rating']:.2f}**."
            )

            st.caption(
                f"Crossover Rate: "
                f"{best_result['Crossover Rate']:.2f} | "
                f"Mutation Rate: "
                f"{best_result['Mutation Rate']:.2f}"
            )

            if (
                results_df["Total Rating"].eq(
                    best_result["Total Rating"]
                ).sum() > 1
            ):
                st.info(
                    "Two or more experiments achieved "
                    "the same highest rating."
                )

        st.markdown("#### Experiment Comparison")

        st.dataframe(
            results_df,
            hide_index=True,
            use_container_width=True
        )

        st.bar_chart(
            results_df.set_index("Experiment")[
                "Total Rating"
            ],
            color="#167A80"
        )

        st.download_button(
            "Download Experiment Comparison",
            data=results_df.to_csv(index=False),
            file_name="experiment_comparison.csv",
            mime="text/csv"
        )


        # =============================================
        # TV SCHEDULE TABLES
        # =============================================

        st.divider()

        st.subheader("05 / TV Programming Schedules")

        st.markdown(
            '<div class="section-caption">'
            'Explore the generated hourly TV timetable '
            'for each experiment.'
            '</div>',
            unsafe_allow_html=True
        )

        tabs = st.tabs([
            "Experiment 1",
            "Experiment 2",
            "Experiment 3"
        ])

        for index, tab in enumerate(tabs):
            with tab:
                name = f"Experiment {index + 1}"
                schedule_df = schedules[name]

                co_rate, mut_rate = (
                    experiment_parameters[index]
                )

                with st.container(border=True):
                    st.markdown(
                        f"#### {name} Schedule"
                    )

                    st.caption(
                        f"Crossover: {co_rate:.2f} | "
                        f"Mutation: {mut_rate:.2f}"
                    )

                    st.dataframe(
                        schedule_df,
                        hide_index=True,
                        use_container_width=True,
                        height=500
                    )

                    total_rating = (
                        results_df.iloc[index][
                            "Total Rating"
                        ]
                    )

                    st.metric(
                        "Total Schedule Rating",
                        f"{total_rating:.2f}"
                    )

                    st.download_button(
                        "Download Schedule CSV",
                        data=schedule_df.to_csv(
                            index=False
                        ),
                        file_name=(
                            f"experiment_{index + 1}"
                            "_schedule.csv"
                        ),
                        mime="text/csv",
                        key=f"download_schedule_{index}"
                    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        TV Scheduling Optimization System
        <br>
        Powered by Genetic Algorithm · Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
