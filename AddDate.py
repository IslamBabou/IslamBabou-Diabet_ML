import pandas as pd
import numpy as np

np.random.seed(42)

def add_date_ddmmyyyy(input_file, output_file, start_date, end_date):
    df = pd.read_csv(input_file)

    # Convertir les dates
    start = pd.to_datetime(start_date, format="%d/%m/%Y")
    end = pd.to_datetime(end_date, format="%d/%m/%Y")

    # Générer décalages en jours
    random_days = np.random.randint(0, (end - start).days + 1, df.shape[0])

    # Générer dates réelles
    generated_dates = start + pd.to_timedelta(random_days, unit="D")

    # Convertir chaque date en string dd/mm/yyyy
    df["Date"] = [d.strftime("%d/%m/%Y") for d in generated_dates]

    df.to_csv(output_file, index=False)
    print(f"✔️ Date added to {output_file}")

# 1️⃣ PIDD
add_date_ddmmyyyy(
    "PIDD.csv",
    "PIDD_with_date.csv",
    "01/01/2023",
    "31/12/2023"
)

# 2️⃣ DataBase-Diabete
add_date_ddmmyyyy(
    "BASEDIABET.csv",
    "BASEDIABET_with_date.csv",
    "01/01/2024",
    "31/12/2024"
)

pidd = pd.read_csv("PIDD_with_date.csv")
pidd_clean = pidd[[
    "Age",
    "BMI",
    "Glucose",
    "BloodPressure",
    "Insulin",
    "Outcome",
    "Date"
]]

pidd_clean.to_csv("PIDD_1.csv", index=False)
print("✔️ PIDD_1.csv created")

# 2️⃣ Nettoyage du dataset local diabète
localdb = pd.read_csv("BASEDIABET_with_date.csv")

localdb_clean = localdb[[
    "age",
    "taille",
    "poids",
    "bmi",
    "gaj",
    "hba1c",
    "type_diabete",
    "Date"
]]

localdb_clean.to_csv("BASEDIABET_1.csv", index=False)
print("✔️ BASEDIABET_1.csv created")
