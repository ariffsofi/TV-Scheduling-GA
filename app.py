
import streamlit as st
import pandas as pd
import random
import io

import ga_algorithm as ga

st.set_page_config(
    page_title="TV Scheduling - Genetic Algorithm",
    page_icon="📺",
    layout="wide"
)

st.title("📺 TV Scheduling Using Genetic Algorithm")
st.write(
    "Optimize TV program scheduling using Genetic Algorithm "
    "and compare three experiments."
)

# Convert CSV data into the required GA format
def prepare_ratings(df):
    if df.shape[1] != 19:
        raise ValueError(
            "CSV must contain a program column and 18 rating columns."
        )

    programs = df.iloc[:, 0].astype(str).str.strip()

    if programs.eq("").any() or programs.duplicated().any():
        raise ValueError("Program names must be unique and non-empty.")

    numeric = df.iloc[:, 1:].apply(
        pd.to_numeric, errors="raise"
    )

    if numeric.isna().any().any():
        raise ValueError("Ratings cannot contain empty values.")

    if not numeric.ge(0).all().all() or not numeric.le(1).all().all():
        raise ValueError("All ratings must be between 0 and 1.")

    return {
        program: [float(value) for value in row]
        for program, row in zip(
            programs, numeric.to_numpy().tolist()
        )
    }


def schedule_to_dataframe(schedule):
    rows = []

    for index, program in enumerate(schedule):
        start = ga.all_time_slots[index]
        end = (start + 1) % 24

        rows.append({
            "Time": f"{start:02d}:00 - {end:02d}:00",
            "TV Program": program.replace("_", " ").title(),
            "Rating": ga.ratings[program][index]
        })

    return pd.DataFrame(rows)


st.subheader("1. Program Ratings Dataset")

uploaded_file = st.file_uploader(
    "Upload your modified program ratings CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        ratings_df = pd.read_csv(uploaded_file)
        new_ratings = prepare_ratings(ratings_df)
        st.success("CSV uploaded and validated successfully.")
    except Exception as error:
        st.error(f"Invalid CSV: {error}")
        st.stop()
else:
    ratings_df = pd.read_csv("program_ratings_modified.csv")
    new_ratings = prepare_ratings(ratings_df)
    st.info("Using the default modified ratings CSV.")

ga.update_ratings(new_ratings)

with st.expander("View Program Ratings"):
    st.dataframe(ratings_df, hide_index=True)

st.subheader("2. Genetic Algorithm Parameters")

st.caption(
    "Set different crossover and mutation rates for each trial."
)

default_rates = [
    (0.80, 0.20),
    (0.70, 0.01),
    (0.95, 0.05)
]

experiment_parameters = []

for i, (default_co, default_mut) in enumerate(default_rates):
    st.markdown(f"**Experiment {i + 1}**")
    col1, col2 = st.columns(2)

    with col1:
        co_rate = st.number_input(
            "Crossover Rate (CO_R)",
            min_value=0.0,
            max_value=0.95,
            value=default_co,
            step=0.05,
            format="%.2f",
            key=f"co_{i}"
        )

    with col2:
        mut_rate = st.number_input(
            "Mutation Rate (MUT_R)",
            min_value=0.01,
            max_value=0.20,
            value=default_mut,
            step=0.01,
            format="%.2f",
            key=f"mut_{i}"
        )

    experiment_parameters.append((co_rate, mut_rate))

st.caption(
    "Mutation rate 0.20 is retained as the assignment default. "
    "Values 0.01–0.05 are also supported."
)

st.subheader("3. Run Genetic Algorithm")

if st.button("Run Three Experiments", type="primary"):
    results = []
    schedules = {}

    for i, (co_rate, mut_rate) in enumerate(experiment_parameters):
        # Reproducible trials
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
            "CO_R": co_rate,
            "MUT_R": mut_rate,
            "Total Rating": round(total_rating, 2)
        })

        schedules[name] = schedule_to_dataframe(schedule)

    st.session_state["results"] = results
    st.session_state["schedules"] = schedules
    st.session_state["dataset_id"] = (
        uploaded_file.getvalue()
        if uploaded_file is not None
        else ratings_df.to_csv(index=False).encode()
    )
    st.session_state["parameters"] = experiment_parameters.copy()

current_dataset = (
    uploaded_file.getvalue()
    if uploaded_file is not None
    else ratings_df.to_csv(index=False).encode()
)

if "results" in st.session_state:
    if (
        st.session_state["dataset_id"] != current_dataset
        or st.session_state["parameters"] != experiment_parameters
    ):
        st.warning(
            "Inputs have changed. Run the experiments again "
            "to update the results."
        )
    else:
        st.subheader("4. Experiment Comparison")

        results_df = pd.DataFrame(st.session_state["results"])
        st.dataframe(results_df, hide_index=True)

        best = results_df.loc[
            results_df["Total Rating"].idxmax()
        ]

        st.success(
            f"Best result: {best['Experiment']} "
            f"with total rating {best['Total Rating']:.2f}"
        )

        st.subheader("5. TV Programming Schedules")

        for name, schedule_df in st.session_state["schedules"].items():
            st.markdown(f"### {name}")
            st.dataframe(
                schedule_df,
                hide_index=True,
                use_container_width=True
            )

            st.download_button(
                f"Download {name} Schedule",
                data=schedule_df.to_csv(index=False),
                file_name=f"{name.replace(' ', '_')}_schedule.csv",
                mime="text/csv",
                key=f"download_{name}"
            )
