import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# === 1️⃣ CHARGEMENT ===
print("📥 Chargement des datasets...")
pidd = pd.read_csv("Partie2/PIDD_preprocessed.csv")
localdb = pd.read_csv("Partie2/LOCAL_preprocessed.csv")

# === 2️⃣ PRÉPARATION ===
print("\n🔧 Préparation des datasets...")

# PIDD (n'a PAS HbA1c, a Insulin)
pidd_prep = pidd.rename(columns={
    'Age': 'age', 'BMI': 'bmi', 'Glucose': 'glucose', 'Insulin': 'insulin'
})[['age', 'bmi', 'glucose', 'insulin', 'label', 'Date']].copy()
pidd_prep['source'] = 'PIDD'

# BASEDIABET (a HbA1c, n'a PAS Insulin)
localdb_prep = localdb.rename(columns={
    'gaj': 'glucose', 'hba1c': 'hba1c'
})[['age', 'bmi', 'glucose', 'hba1c', 'label', 'Date']].copy()
localdb_prep['source'] = 'BASEDIABET'

print("✅ Datasets préparés")


# === 3️⃣ PRÉDICTION HbA1c POUR PIDD  ===

print("\n PRÉDICTION HbA1c POUR PIDD ...")

# BASEDIABET a les vraies valeurs HbA1c → on l'utilise pour entraîner
X_hba1c_train = localdb_prep[['age', 'bmi', 'glucose']].copy()
y_hba1c_train = localdb_prep['hba1c'].copy()  # ← Vraies valeurs

# Nettoyage
X_hba1c_train = X_hba1c_train.fillna(X_hba1c_train.median())
y_hba1c_train = y_hba1c_train.fillna(y_hba1c_train.median())

# Entraînement modèle HbA1c
hba1c_predictor = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
hba1c_predictor.fit(X_hba1c_train, y_hba1c_train)

# Prédiction HbA1c  pour PIDD
X_pidd_hba1c = pidd_prep[['age', 'bmi', 'glucose']].copy()
X_pidd_hba1c = X_pidd_hba1c.fillna(X_pidd_hba1c.median())
pidd_hba1c_pred = hba1c_predictor.predict(X_pidd_hba1c)

#  On ajoute HbA1c SEULEMENT à PIDD
pidd_prep['hba1c'] = pidd_hba1c_pred  # ← Prédit pour PIDD

print(f"📊 HbA1c - PIDD (prédit): {pidd_hba1c_pred.mean():.2f}, BASEDIABET (réel): {localdb_prep['hba1c'].mean():.2f}")

# === 4️⃣ PRÉDICTION INSULIN POUR BASEDIABET SEULEMENT ===
print("\n💉 PRÉDICTION INSULIN POUR BASEDIABET SEULEMENT...")

# PIDD a les vraies valeurs Insulin → on l'utilise pour entraîner
X_insulin_train = pidd_prep[['age', 'bmi', 'glucose', 'hba1c']].copy()
y_insulin_train = pidd_prep['insulin'].copy()  # ← Vraies valeurs

# Nettoyage
X_insulin_train = X_insulin_train.fillna(X_insulin_train.median())
y_insulin_train = y_insulin_train.fillna(y_insulin_train.median())

# Entraînement modèle Insulin
insulin_predictor = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
insulin_predictor.fit(X_insulin_train, y_insulin_train)

# Prédiction Insulin UNIQUEMENT pour BASEDIABET
X_basediabet_insulin = localdb_prep[['age', 'bmi', 'glucose', 'hba1c']].copy()
X_basediabet_insulin = X_basediabet_insulin.fillna(X_basediabet_insulin.median())
localdb_insulin_pred = insulin_predictor.predict(X_basediabet_insulin)

# ⭐ IMPORTANT : On ajoute Insulin SEULEMENT à BASEDIABET
localdb_prep['insulin'] = localdb_insulin_pred  # ← Prédit pour BASEDIABET
# pidd_prep garde son Insulin ORIGINAL ✅

print(f"📊 Insulin - PIDD (réel): {pidd_prep['insulin'].mean():.1f}, BASEDIABET (prédit): {localdb_insulin_pred.mean():.1f}")

# === 5️⃣ FUSION FINALE ===
print("\n🔄 FUSION DES DATASETS...")

# Colonnes FINALES (sans colonnes source inutiles)
common_columns = ['age', 'bmi', 'glucose', 'hba1c', 'insulin', 'label', 'Date', 'source']

# Fusion
combined_final = pd.concat([
    pidd_prep[common_columns],
    localdb_prep[common_columns]
], ignore_index=True)

print(f"✅ Dataset fusionné final: {combined_final.shape}")

# === 6️⃣ VÉRIFICATION ===
print("\n✅ VÉRIFICATION DES VALEURS:")

# Vérifier que BASEDIABET garde ses vraies valeurs HbA1c
hba1c_basediabet_original = localdb['hba1c'].mean()  # Original
hba1c_basediabet_final = combined_final[combined_final['source']=='BASEDIABET']['hba1c'].mean()

print(f"BASEDIABET HbA1c - Original: {hba1c_basediabet_original:.2f}, Final: {hba1c_basediabet_final:.2f} → {'✅ Identique' if abs(hba1c_basediabet_original - hba1c_basediabet_final) < 0.01 else '❌ Modifié'}")

# Vérifier que PIDD garde ses vraies valeurs Insulin
insulin_pidd_original = pidd['Insulin'].mean()  # Original
insulin_pidd_final = combined_final[combined_final['source']=='PIDD']['insulin'].mean()

print(f"PIDD Insulin - Original: {insulin_pidd_original:.1f}, Final: {insulin_pidd_final:.1f} → {'✅ Identique' if abs(insulin_pidd_original - insulin_pidd_final) < 0.1 else '❌ Modifié'}")

# === 7️⃣ SAUVEGARDE ===
print("\n💾 SAUVEGARDE...")
combined_final.to_csv('Partie2/dataset_fusionne_complet.csv', index=False)

print("""
🎉 FUSION TERMINÉE !

📊 RÉSULTAT FINAL:
- PIDD: HbA1c prédit 
- BASEDIABET: Insulin prédit

📁 Fichier: dataset_fusionne_complet.csv
✅ Prêt pour le Machine Learning !
""")