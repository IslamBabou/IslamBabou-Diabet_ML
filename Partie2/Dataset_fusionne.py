import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE, ADASYN
import warnings
warnings.filterwarnings('ignore')

# === 1️⃣ CHARGEMENT DES DONNÉES FUSIONNÉES ===
print("📥 Chargement des données fusionnées...")
DATASET_FUSIONNE = pd.read_csv("Partie2/dataset_fusionne_complet.csv")

print("🔍 Aperçu des données :")
print(DATASET_FUSIONNE.head())
print(f"\n📊 Shape: {DATASET_FUSIONNE.shape}")

# === 2️⃣ SUPPRESSION DES COLONNES DATE ET SOURCE ===
print("🔧 Suppression des colonnes Date et Source...")

# Vérifier quelles colonnes existent dans le dataset
colonnes_existantes = DATASET_FUSIONNE.columns.tolist()
print(f"📋 Colonnes disponibles: {colonnes_existantes}")

# Liste des colonnes à supprimer
colonnes_a_supprimer = []
colonnes_a_supprimer.append('Date')
colonnes_a_supprimer.append('source')

print(f"🗑️ Colonnes à supprimer: {colonnes_a_supprimer}")

# Supprimer les colonnes problématiques + Date et Source
X_FUSIONNE = DATASET_FUSIONNE.drop(columns=['label'] + colonnes_a_supprimer)
y_FUSIONNE = DATASET_FUSIONNE['label']

print(f"✅ Features après suppression: {X_FUSIONNE.columns.tolist()}")
print(f"📊 Types des features: {X_FUSIONNE.dtypes}")

# Vérifier s'il reste des colonnes non-numériques problématiques
non_numeric_cols = X_FUSIONNE.select_dtypes(include=['object']).columns
if len(non_numeric_cols) > 0:
    print(f"⚠️  Colonnes non-numériques restantes: {non_numeric_cols.tolist()}")
    print("🔧 Suppression des colonnes non-numériques restantes...")
    X_FUSIONNE = X_FUSIONNE.drop(columns=non_numeric_cols)
    print(f"✅ Features finales: {X_FUSIONNE.columns.tolist()}")

# === 3️⃣ DIVISION TRAIN/TEST ===
X_train_FUSIONNE, X_test_FUSIONNE, y_train_FUSIONNE, y_test_FUSIONNE = train_test_split(
    X_FUSIONNE, y_FUSIONNE, test_size=0.3, random_state=42, stratify=y_FUSIONNE
)

print(f"📈 Distribution - Train: {np.bincount(y_train_FUSIONNE)}, Test: {np.bincount(y_test_FUSIONNE)}")

# === 4️⃣ DÉFINITION DES MODÈLES ===
models = {
    'XGBoost': XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        scale_pos_weight=np.sum(y_train_FUSIONNE == 0) / np.sum(y_train_FUSIONNE == 1)
    ),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
    'Naive Bayes': GaussianNB()
}

sampling_methods = {
    'Original': None,
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42)
}

# === 5️⃣ ENTRAÎNEMENT ET ÉVALUATION ===
results = {}

for sample_name, sampler in sampling_methods.items():
    print(f"\n{'='*50}")
    print(f"🎯 MÉTHODE D'ÉCHANTILLONNAGE: {sample_name}")
    print(f"{'='*50}")
    
    # Préparer les données
    if sampler:
        X_resampled, y_resampled = sampler.fit_resample(X_train_FUSIONNE, y_train_FUSIONNE)
        print(f"Distribution après {sample_name}: {np.bincount(y_resampled)}")
    else:
        X_resampled, y_resampled = X_train_FUSIONNE, y_train_FUSIONNE
    
    results[sample_name] = {}
    
    for model_name, model in models.items():
        print(f"\n🏃‍♂️ Entraînement {model_name}...")
        
        try:
            model.fit(X_resampled, y_resampled)
            y_pred = model.predict(X_test_FUSIONNE)
            y_proba = model.predict_proba(X_test_FUSIONNE)[:, 1]
            
            report = classification_report(y_test_FUSIONNE, y_pred, output_dict=True)
            auc = roc_auc_score(y_test_FUSIONNE, y_proba)
            
            results[sample_name][model_name] = {
                'model': model,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'precision_0': report['0']['precision'],
                'recall_0': report['0']['recall'],
                'f1_0': report['0']['f1-score'],
                'precision_1': report['1']['precision'],
                'recall_1': report['1']['recall'],
                'f1_1': report['1']['f1-score'],
                'auc': auc,
                'support_0': report['0']['support'],
                'support_1': report['1']['support']
            }
            
            print(f"✅ {model_name} - AUC: {auc:.3f}, F1-Score Classe 1: {report['1']['f1-score']:.3f}")
            
        except Exception as e:
            print(f"❌ Erreur avec {model_name}: {e}")
            results[sample_name][model_name] = None



# === 5️⃣ VISUALISATIONS COMPLÈTES ===
print("\n📊 Génération des visualisations...")

# Configuration des styles
plt.style.use('default')
sns.set_palette("husl")

# === 5.1 COMPARAISON DES PERFORMANCES ===
fig, axes = plt.subplots(2, 2, figsize=(22, 16))

# Métriques à comparer
metrics = ['precision_1', 'recall_1', 'f1_1', 'auc']
metric_names = ['Précision (Classe 1)', 'Rappel (Classe 1)', 'F1-Score (Classe 1)', 'AUC-ROC']

for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx//2, idx%2]
    
    # Préparation des données pour le graphique
    data = []
    for sample_name in sampling_methods.keys():
        for model_name in models.keys():
            if results[sample_name][model_name] is not None:
                value = results[sample_name][model_name][metric]
                data.append({'Méthode': sample_name, 'Modèle': model_name, 'Valeur': value})
    
    df_plot = pd.DataFrame(data)
    
    # Graphique barres
    if not df_plot.empty:
        sns.barplot(data=df_plot, x='Méthode', y='Valeur', hue='Modèle', ax=ax)
        ax.set_title(f'Comparaison des Modèles - {name}', fontsize=14, fontweight='bold')
        ax.set_ylabel(name)
        ax.set_ylim(0, 1)
        
        # Ajout des valeurs sur les barres
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
        
        # Rotation des labels pour meilleure lisibilité
        ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('Partie2/Fusionne_results/comparaison_performances.png', dpi=300, bbox_inches='tight')

# === 5.2 COURBES ROC ===
plt.figure(figsize=(14, 8))

for sample_name in sampling_methods.keys():
    for model_name in models.keys():
        if results[sample_name][model_name] is not None:
            y_proba = results[sample_name][model_name]['y_proba']
            fpr, tpr, _ = roc_curve(y_test_FUSIONNE, y_proba)  # ⬅️ CORRECTION ICI
            auc = results[sample_name][model_name]['auc']
            
            plt.plot(fpr, tpr, label=f'{sample_name} + {model_name} (AUC = {auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Aléatoire')
plt.xlabel('Taux de Faux Positifs', fontsize=12)
plt.ylabel('Taux de Vrais Positifs', fontsize=12)
plt.title('Courbes ROC - Comparaison des Modèles', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Partie2/Fusionne_results/courbes_roc.png', dpi=300, bbox_inches='tight')

# === 5.3 MATRICES DE CONFUSION ===
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

for i, sample_name in enumerate(sampling_methods.keys()):
    for j, model_name in enumerate(models.keys()):
        ax = axes[i, j]
        
        if results[sample_name][model_name] is not None:
            y_pred = results[sample_name][model_name]['y_pred']
            cm = confusion_matrix(y_test_FUSIONNE, y_pred)  # ⬅️ CORRECTION ICI
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Sain', 'Diabétique'],
                       yticklabels=['Sain', 'Diabétique'])
            
            ax.set_title(f'{sample_name} + {model_name}\nF1: {results[sample_name][model_name]["f1_1"]:.3f}')
            ax.set_xlabel('Prédit')
            ax.set_ylabel('Réel')
        else:
            ax.text(0.5, 0.5, 'Erreur', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{sample_name} + {model_name}')

plt.tight_layout()
plt.savefig('Partie2/Fusionne_results/matrices_confusion.png', dpi=300, bbox_inches='tight')


# === 5.4 RAPPEL vs PRÉCISION PAR CLASSE ===
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# Classe 0 (Sain)
ax1 = axes[0]
data_class0 = []
for sample_name in sampling_methods.keys():
    for model_name in models.keys():
        if results[sample_name][model_name] is not None:
            data_class0.append({
                'Méthode': f"{sample_name}\n{model_name}",
                'Précision': results[sample_name][model_name]['precision_0'],
                'Rappel': results[sample_name][model_name]['recall_0']
            })

df_class0 = pd.DataFrame(data_class0)
x = range(len(df_class0))
width = 0.35

ax1.bar([i - width/2 for i in x], df_class0['Précision'], width, label='Précision', alpha=0.8)
ax1.bar([i + width/2 for i in x], df_class0['Rappel'], width, label='Rappel', alpha=0.8)
ax1.set_title('Performance - Classe 0 (Sains)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Score')
ax1.set_xticks(x)
ax1.set_xticklabels(df_class0['Méthode'], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Classe 1 (Diabétique)
ax2 = axes[1]
data_class1 = []
for sample_name in sampling_methods.keys():
    for model_name in models.keys():
        if results[sample_name][model_name] is not None:
            data_class1.append({
                'Méthode': f"{sample_name}\n{model_name}",
                'Précision': results[sample_name][model_name]['precision_1'],
                'Rappel': results[sample_name][model_name]['recall_1']
            })

df_class1 = pd.DataFrame(data_class1)
x = range(len(df_class1))

ax2.bar([i - width/2 for i in x], df_class1['Précision'], width, label='Précision', alpha=0.8)
ax2.bar([i + width/2 for i in x], df_class1['Rappel'], width, label='Rappel', alpha=0.8)
ax2.set_title('Performance - Classe 1 (Diabétiques)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Score')
ax2.set_xticks(x)
ax2.set_xticklabels(df_class1['Méthode'], rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Partie2/Fusionne_results/precision_rappel_par_classe.png', dpi=300, bbox_inches='tight')

# === 6️⃣ RAPPORT DÉTAILLÉ DES PERFORMANCES ===
print("\n" + "="*60)
print("📈 RAPPORT FINAL DES PERFORMANCES")
print("="*60)

performance_data = []

for sample_name in sampling_methods.keys():
    for model_name in models.keys():
        if results[sample_name][model_name] is not None:
            perf = results[sample_name][model_name]
            performance_data.append({
                'Méthode': sample_name,
                'Modèle': model_name,
                'AUC': f"{perf['auc']:.3f}",
                
                # === METRIQUES CLASSE 0 (SAINS) ===
                'Précision Classe 0': f"{perf['precision_0']:.3f}",
                'Rappel Classe 0': f"{perf['recall_0']:.3f}",
                'F1-Score Classe 0': f"{perf['f1_0']:.3f}",
                
                # === METRIQUES CLASSE 1 (DIABÉTIQUES) ===
                'Précision Classe 1': f"{perf['precision_1']:.3f}",
                'Rappel Classe 1': f"{perf['recall_1']:.3f}",
                'F1-Score Classe 1': f"{perf['f1_1']:.3f}"
            })

df_performance = pd.DataFrame(performance_data)
print("\nTableau récapitulatif des performances :")
print(df_performance.to_string(index=False))

# Sauvegarde avec encodage correct
df_performance.to_csv('Partie2/Fusionne_results/tableau_performances.csv', index=False, encoding='utf-8-sig')

# === 7️⃣ IDENTIFICATION DU MEILLEUR MODÈLE ===
best_f1 = 0
best_combo = None

for sample_name in sampling_methods.keys():
    for model_name in models.keys():
        if results[sample_name][model_name] is not None:
            f1_score = results[sample_name][model_name]['f1_1']
            if f1_score > best_f1:
                best_f1 = f1_score
                best_combo = (sample_name, model_name)

print(f"\n🏆 MEILLEUR MODÈLE: {best_combo[0]} + {best_combo[1]}")
print(f"🎯 F1-Score Classe 1: {best_f1:.3f}")

print("\n✅ Toutes les visualisations ont été sauvegardées dans le dossier 'Partie2/Fusionne_results/'")
print("📊 Fichiers générés:")
print("   - comparaison_performances.png")
print("   - courbes_roc.png")
print("   - matrices_confusion.png")
print("   - precision_rappel_par_classe.png")
print("   - tableau_performances.csv")