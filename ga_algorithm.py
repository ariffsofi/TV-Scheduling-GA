
import csv
import random

# Default Genetic Algorithm parameters
GEN = 100
POP = 50
CO_R = 0.8
MUT_R = 0.2
EL_S = 2

all_time_slots = list(range(6, 24))

# Read program ratings from CSV
def read_csv_to_dict(file_path):
    program_ratings = {}

    with open(file_path, mode="r", newline="") as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            program = row[0]
            ratings_list = [float(x) for x in row[1:]]
            program_ratings[program] = ratings_list

    return program_ratings


# Load default modified ratings
ratings = read_csv_to_dict("program_ratings_modified.csv")
all_programs = list(ratings.keys())


# Update ratings when a CSV is uploaded
def update_ratings(new_ratings):
    global ratings, all_programs

    if not new_ratings:
        raise ValueError("The ratings dataset is empty.")

    if any(len(values) != len(all_time_slots)
           for values in new_ratings.values()):
        raise ValueError(
            "Each program must have 18 hourly ratings."
        )

    if any(
        not 0 <= value <= 1
        for values in new_ratings.values()
        for value in values
    ):
        raise ValueError("Ratings must be between 0 and 1.")

    ratings = new_ratings
    all_programs = list(ratings.keys())


# Fitness function
def fitness_function(schedule):
    total_rating = 0

    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]

    return total_rating


# Initialize population
def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]

    all_schedules = []

    for i in range(len(programs)):
        for schedule in initialize_pop(
            programs[:i] + programs[i + 1:], time_slots
        ):
            all_schedules.append(
                [programs[i]] + schedule
            )

    return all_schedules


# Find best schedule
def finding_best_schedule(all_schedules):
    return max(
        all_schedules,
        key=fitness_function
    )


# Single-point crossover
def crossover(schedule1, schedule2):
    crossover_point = random.randint(
        1, len(schedule1) - 2
    )

    child1 = (
        schedule1[:crossover_point]
        + schedule2[crossover_point:]
    )

    child2 = (
        schedule2[:crossover_point]
        + schedule1[crossover_point:]
    )

    return child1, child2


# Mutation
def mutate(schedule):
    mutation_point = random.randint(
        0, len(schedule) - 1
    )

    new_program = random.choice(all_programs)

    schedule[mutation_point] = new_program

    return schedule


def evaluate_fitness(schedule):
    return fitness_function(schedule)


# Genetic Algorithm
def genetic_algorithm(
    initial_schedule=None,
    generations=GEN,
    population_size=POP,
    crossover_rate=CO_R,
    mutation_rate=MUT_R,
    elitism_size=EL_S
):

    if initial_schedule is None:
        initial_schedule = [
            random.choice(all_programs)
            for _ in all_time_slots
        ]

    if len(initial_schedule) != len(all_time_slots):
        raise ValueError(
            "The initial schedule must have 18 programs."
        )

    population = [initial_schedule.copy()]

    for _ in range(population_size - 1):
        random_schedule = [
            random.choice(all_programs)
            for _ in all_time_slots
        ]

        population.append(random_schedule)

    for generation in range(generations):
        new_population = []

        # Elitism
        population.sort(
            key=fitness_function,
            reverse=True
        )

        new_population.extend(
            schedule.copy()
            for schedule in population[:elitism_size]
        )

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(
                population, k=2
            )

            if random.random() < crossover_rate:
                child1, child2 = crossover(
                    parent1, parent2
                )
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1)

            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend(
                [child1, child2]
            )

        population = new_population[:population_size]

    # Return the best schedule
    return max(population, key=fitness_function)
