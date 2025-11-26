import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# === 1️⃣ Charger les fichiers nettoyés ===
pidd = pd.read_csv("PIDD_1.csv")
localdb = pd.read_csv("BASEDIABET_1.csv")

# === 2️⃣ Uniformiser le nom de la colonne label ===
pidd = pidd.rename(columns={"Outcome": "label"})
localdb = localdb.rename(columns={"type_diabete": "label"})

# === 3️⃣ Séparer X (features) et y (label) ===
def split_xy(df):
    X = df.drop(columns=["label", "Date"])  
    y = df["label"]
    return X, y

Xp, yp = split_xy(pidd)
Xl, yl = split_xy(localdb)

# === 4️⃣ Imputation des valeurs manquantes (médiane) ===
imputer = SimpleImputer(strategy="median")

Xp_imputed = pd.DataFrame(imputer.fit_transform(Xp), columns=Xp.columns)
Xl_imputed = pd.DataFrame(imputer.fit_transform(Xl), columns=Xl.columns)

# === 5️⃣ Standardisation ===
scaler = StandardScaler()

Xp_scaled = pd.DataFrame(scaler.fit_transform(Xp_imputed), columns=Xp_imputed.columns)
Xl_scaled = pd.DataFrame(scaler.fit_transform(Xl_imputed), columns=Xl_imputed.columns)

# === 6️⃣ Reconstituer datasets finaux (X normalisé + y label + Date si utile) ===
PIDD_preprocessed = pd.concat([Xp_scaled, yp, pidd["Date"]], axis=1)
LOCAL_preprocessed = pd.concat([Xl_scaled, yl, localdb["Date"]], axis=1)

# === 7️⃣ Sauvegarde ===
PIDD_preprocessed.to_csv("Partie2/PIDD_preprocessed.csv", index=False)
LOCAL_preprocessed.to_csv("Partie2/LOCAL_preprocessed.csv", index=False)

print("✔️ Prétraitement terminé ! Fichiers générés :")
print("- PIDD_preprocessed.csv")
print("- LOCAL_preprocessed.csv")
