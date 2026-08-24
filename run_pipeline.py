"""
Pipeline complet de prédiction du départ client.

    python run_pipeline.py

Enchaîne : chargement, nettoyage, création de variables, comparaison des
modèles par validation croisée, choix du seuil de décision par le coût, et
interprétation. Résultats dans reports/ et figures/.
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_predict)
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, recall_score, precision_score,
                             confusion_matrix)
from sklearn.inspection import permutation_importance

import data
import models
import decision
import plots

RACINE = os.path.dirname(os.path.abspath(__file__))
REPORTS, FIGURES = os.path.join(RACINE, "reports"), os.path.join(RACINE, "figures")
ALEA = 42


def titre(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def main():
    os.makedirs(REPORTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    t_debut = time.time()

    # ------------------------------------------------------------------ 1
    titre("1. Données")
    df = data.enrichir(data.charger())
    print(f"  {len(df)} clients, {df.shape[1] - 1} variables")
    print(f"  taux de départ : {df['Churn'].mean():.1%}")
    print(f"  -> une réponse constante 'ce client reste' donnerait déjà "
          f"{1 - df['Churn'].mean():.1%} d'exactitude")
    plots.profil_risque(df, f"{FIGURES}/profil_risque.png")

    X, y, numeriques, categorielles = data.separer(df)
    print(f"  {len(numeriques)} variables numériques, "
          f"{len(categorielles)} variables catégorielles")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=ALEA)
    print(f"  entraînement : {len(X_train)}   test : {len(X_test)}")

    # ------------------------------------------------------------------ 2
    titre("2. Comparaison des modèles (validation croisée 5 plis)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=ALEA)
    catalogue = models.catalogue(numeriques, categorielles)

    scores, probas_test, modeles_ajustes = [], {}, {}
    for nom, pipeline in catalogue.items():
        oof = cross_val_predict(pipeline, X_train, y_train, cv=cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
        pipeline.fit(X_train, y_train)
        p_test = pipeline.predict_proba(X_test)[:, 1]
        probas_test[nom], modeles_ajustes[nom] = p_test, pipeline

        pred_test = (p_test >= 0.5).astype(int)
        scores.append({
            "modele": nom,
            "auc_validation": round(roc_auc_score(y_train, oof), 4),
            "auc_test": round(roc_auc_score(y_test, p_test), 4),
            "ap_test": round(average_precision_score(y_test, p_test), 4),
            "exactitude_test": round(accuracy_score(y_test, pred_test), 4),
            "rappel_test": round(recall_score(y_test, pred_test), 4),
            "precision_test": round(precision_score(y_test, pred_test, zero_division=0), 4),
        })
        # Mémorise les probabilités hors-échantillon du meilleur modèle
        catalogue[nom] = oof

    resultats = pd.DataFrame(scores).sort_values("auc_test", ascending=False)
    print(resultats.to_string(index=False))
    resultats.to_csv(f"{REPORTS}/comparaison_modeles.csv", index=False)

    meilleur = resultats.iloc[0]["modele"]
    print(f"\n  Modèle retenu : {meilleur}")
    print("  Remarque : l'exactitude est volontairement reléguée en fin de tableau.")
    print("  Avec 26,5 % de départs, elle récompense un modèle qui prédit "
          "'personne ne part'.")

    plots.courbes_roc(probas_test, y_test, f"{FIGURES}/courbes_roc.png")

    # ------------------------------------------------------------------ 3
    titre("3. Choix du seuil de décision par le coût")
    print(f"  hypothèses : client perdu = {decision.COUT_FN:.0f} €, "
          f"geste commercial inutile = {decision.COUT_FP:.0f} €")

    oof_meilleur = catalogue[meilleur]
    seuil, balayage = decision.seuil_optimal(y_train.values, oof_meilleur)
    print(f"  seuil retenu sur la validation croisée : {seuil:.2f} "
          f"(au lieu de 0,50 par défaut)")
    balayage.to_csv(f"{REPORTS}/balayage_seuils.csv", index=False)
    plots.courbe_cout(balayage, seuil, f"{FIGURES}/cout_par_seuil.png")

    comparaison = decision.comparer(y_test.values, probas_test[meilleur], seuil)
    print()
    print(comparaison.to_string(index=False))
    comparaison.to_csv(f"{REPORTS}/impact_economique.csv", index=False)

    ligne_opt = comparaison.iloc[-1]
    ligne_def = comparaison.iloc[-2]
    economie = int(ligne_def["cout_euros"] - ligne_opt["cout_euros"])
    print(f"\n  Le modèle n'a pas changé, seul le seuil a bougé : "
          f"{economie} € économisés sur le jeu de test "
          f"({ligne_opt['gain_vs_defaut_%']:.0f} %)")

    print("\n  Sensibilité à l'hypothèse de coût :")
    sens = decision.sensibilite(y_train.values, oof_meilleur,
                                y_test.values, probas_test[meilleur])
    print(sens.to_string(index=False))
    sens.to_csv(f"{REPORTS}/sensibilite_couts.csv", index=False)

    mc = confusion_matrix(y_test, (probas_test[meilleur] >= seuil).astype(int))
    print(f"\n  Matrice de confusion au seuil retenu :")
    print(f"    reste, non contacté   {mc[0, 0]:>5}   |  reste, contacté à tort {mc[0, 1]:>5}")
    print(f"    part, non détecté     {mc[1, 0]:>5}   |  part, détecté          {mc[1, 1]:>5}")

    # ------------------------------------------------------------------ 4
    titre("4. Sur quoi le modèle s'appuie")
    print("  (importance par permutation : on mélange une variable et on regarde")
    print("   de combien l'AUC chute. Plus fiable que l'importance native des")
    print("   arbres, qui surévalue les variables à nombreuses modalités.)")

    perm = permutation_importance(
        modeles_ajustes[meilleur], X_test, y_test, scoring="roc_auc",
        n_repeats=10, random_state=ALEA, n_jobs=-1)
    imp = (pd.DataFrame({"variable": X_test.columns,
                         "importance": perm.importances_mean,
                         "ecart_type": perm.importances_std})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    print()
    print(imp.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    imp.to_csv(f"{REPORTS}/importances.csv", index=False)
    plots.importances(imp, f"{FIGURES}/importances.png")

    construites = imp[imp["variable"].isin(
        ["nb_services", "facture_moyenne", "evolution_facture", "client_recent"])]
    print(f"\n  Variables construites dans le top 10 : "
          f"{[v for v in construites.head(4)['variable'] if v in list(imp.head(10)['variable'])]}")

    titre(f"Terminé en {time.time() - t_debut:.1f}s — reports/ et figures/")


if __name__ == "__main__":
    main()
