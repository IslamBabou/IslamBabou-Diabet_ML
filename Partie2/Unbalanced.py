import pandas as pd

def class_imbalance_report(df, label_col="label", name="Dataset"):
    total = len(df)
    count_0 = (df[label_col] == 0).sum()
    count_1 = (df[label_col] == 1).sum()

    pct_0 = (count_0 / total) * 100
    pct_1 = (count_1 / total) * 100

    print(f"\n====== {name} ======")
    print(f"Total exemples : {total}")
    print(f"Classe 0 (non diabétique) : {count_0} ({pct_0:.2f}%)")
    print(f"Classe 1 (diabétique)     : {count_1} ({pct_1:.2f}%)")
    print(f"Déséquilibre : Classe majoritaire = {'0' if count_0 > count_1 else '1'}")
    print("===========================")

# Charger les datasets
pidd = pd.read_csv(r"C:\Users\USER\OneDrive\Bureau\DW-BDreussite\2\PIDD_1.csv")
local = pd.read_csv(r"C:\Users\USER\OneDrive\Bureau\DW-BDreussite\2\BASEDIABET_1.csv")



# Renommer label pour uniformité si nécessaire
pidd = pidd.rename(columns={"Outcome": "label"})
local = local.rename(columns={"type_diabete": "label"}) 

# Calcul du déséquilibre
class_imbalance_report(pidd, label_col="label", name="PIDD Dataset")
class_imbalance_report(local, label_col="label", name="Local Diabetes Dataset")
