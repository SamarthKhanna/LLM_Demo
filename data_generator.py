import pandas as pd
import random

# Example employee templates
names = [
    ("Alice", "highly consistent performer, exceeds targets, mentors juniors, fosters team morale"),
    ("Bob", "reliable worker, meets expectations, but rarely takes initiative"),
    ("Carol", "new employee with great potential, adaptable, and eager to learn"),
    ("Dan", "experienced but performance has declined in recent quarters"),
    ("Eva", "creative problem-solver, brings innovative ideas, but sometimes misses deadlines"),
    ("Frank", "technically skilled, quiet, delivers solid results but lacks leadership"),
    ("Grace", "high performer, recently promoted, leads key projects"),
    ("Henry", "steady contributor, good attendance, avoids conflicts"),
    ("Irene", "detail-oriented and dependable, though sometimes struggles with time management"),
    ("Jack", "very ambitious, strong communicator, sometimes overconfident"),
]

# Helper function to create 4 distinct employees
def make_employee_group(seed):
    random.seed(seed)
    selected = random.sample(names, 4)
    employees = []
    for i, (name, desc) in enumerate(selected, start=1):
        employees.append(f"{chr(64+i)}: {name} – {desc}.")
    return employees

# Create 20 examples
examples = []
for i in range(1, 21):
    employees = make_employee_group(i)
    examples.append({
        "example_id": i,
        "employee_A": employees[0],
        "employee_B": employees[1],
        "employee_C": employees[2],
        "employee_D": employees[3]
    })

df = pd.DataFrame(examples)

# Save to CSV
csv_path = "employee_comparison_examples.csv"
df.to_csv(csv_path, index=False)

csv_path
