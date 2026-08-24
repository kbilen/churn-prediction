"""
Chargement, nettoyage et création de variables.

Le nettoyage tient en peu de lignes mais contient le piège classique de ce jeu
de données : la colonne TotalCharges est lue comme du texte à cause de 11
valeurs vides. Ces 11 clients ne sont pas des erreurs de saisie, ce sont des
abonnés dont l'ancienneté est nulle : ils viennent de souscrire et n'ont donc
encore rien facturé. Les supprimer ferait perdre une information (le client
tout neuf), les remplacer par 0 est le comportement correct.
"""

import os
import pandas as pd

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMIN = os.path.join(RACINE, "data", "telco_churn.csv")

SERVICES = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies"]

CIBLE = "Churn"


def charger():
    df = pd.read_csv(CHEMIN)

    # TotalCharges : texte à cause des 11 valeurs vides -> numérique
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df[CIBLE] = (df[CIBLE] == "Yes").astype(int)
    df = df.drop(columns=["customerID"])
    return df


def enrichir(df):
    """
    Quatre variables construites, chacune avec une raison métier.

    On ne fabrique pas des variables pour faire du volume : chaque ajout doit
    correspondre à une hypothèse sur le comportement du client.
    """
    df = df.copy()

    # 1. Nombre de services souscrits : plus un client est équipé, plus il est
    #    coûteux pour lui de partir ailleurs.
    def compte_services(ligne):
        n = 0
        for c in SERVICES:
            v = ligne[c]
            if v not in ("No", "No internet service", "No phone service"):
                n += 1
        return n

    df["nb_services"] = df.apply(compte_services, axis=1)

    # 2. Facture moyenne réellement observée. Peut différer de MonthlyCharges
    #    si le client a changé d'offre en cours de route.
    df["facture_moyenne"] = df["TotalCharges"] / df["tenure"].clip(lower=1)

    # 3. Écart entre la facture actuelle et la moyenne historique : une hausse
    #    récente est un motif de départ fréquent.
    df["evolution_facture"] = df["MonthlyCharges"] - df["facture_moyenne"]

    # 4. Client récent : le risque de départ est concentré sur les premiers mois.
    df["client_recent"] = (df["tenure"] <= 6).astype(int)

    return df


def separer(df):
    X = df.drop(columns=[CIBLE])
    y = df[CIBLE]
    numeriques = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorielles = [c for c in X.columns if c not in numeriques]
    return X, y, numeriques, categorielles
