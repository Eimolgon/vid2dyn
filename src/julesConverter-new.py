
import pandas as pd
from pathlib import Path

source = Path("julesFiltered_gopro_test_1_1.csv")
destination = Path("states_jf_gopro_test_1_1.csv")

df = pd.read_csv(source)

converted = pd.DataFrame({
    "time": df["time"],
    "q1": df["x"],
    "q2": df["y"],
    "q3": df["psi"],
    "q4": df["phi"],
    "q7": df["delta"],
    "u3": df["psi_dot"],
    "u4": df["phi_dot"],
    "u7": df["delta_dot"],
    "speed": df["v"],
})

destination.parent.mkdir(
    parents=True,
    exist_ok=True
)

converted.to_csv(destination, index=False)

print("Converted dataset:", destination)
print(converted.head())
print("Samples:", len(converted))
