"""
Du score de probabilité à la décision.

Un modèle de classification ne rend pas une décision, il rend une probabilité.
C'est le seuil qui transforme cette probabilité en action : appeler ce client,
ou ne pas l'appeler. Par défaut, scikit-learn utilise 0,50 — un choix qui n'a
aucune justification métier et qui suppose implicitement que se tromper dans un
sens coûte autant que se tromper dans l'autre.

Ici ce n'est pas le cas, et c'est tout l'enjeu :

  - laisser partir un client sans rien faire (faux négatif) coûte la valeur
    du contrat perdu, soit environ 500 €
  - offrir un geste commercial à un client qui serait resté de toute façon
    (faux positif) coûte le geste lui-même, soit environ 50 €

Une erreur coûte dix fois l'autre. Le bon seuil n'est donc pas 0,50, il est
bien plus bas : on préfère largement appeler quelques clients pour rien plutôt
que d'en laisser filer un seul.

Point de méthode important : le seuil est choisi sur des prédictions
hors-échantillon obtenues par validation croisée sur le jeu d'entraînement,
puis appliqué tel quel au jeu de test. Le choisir directement sur le test
reviendrait à optimiser sur les données d'évaluation, et le gain annoncé serait
fictif.
"""

import numpy as np
import pandas as pd

COUT_FN = 500.0   # client perdu faute d'avoir été identifié
COUT_FP = 50.0    # geste commercial, payé pour chaque client contacté


def cout_total(y_vrai, y_pred, cout_fn=COUT_FN, cout_fp=COUT_FP):
    fn = int(((y_vrai == 1) & (y_pred == 0)).sum())
    fp = int(((y_vrai == 0) & (y_pred == 1)).sum())
    contactes = int((y_pred == 1).sum())
    return fn * cout_fn + contactes * cout_fp, fn, fp


def balayer(y_vrai, proba, cout_fn=COUT_FN, cout_fp=COUT_FP, n=201):
    """Coût total pour chaque seuil possible entre 0,01 et 0,99."""
    lignes = []
    for s in np.linspace(0.01, 0.99, n):
        y_pred = (proba >= s).astype(int)
        cout, fn, fp = cout_total(y_vrai, y_pred, cout_fn, cout_fp)
        lignes.append({"seuil": round(float(s), 4), "cout": cout,
                       "faux_negatifs": fn, "faux_positifs": fp,
                       "clients_contactes": int(y_pred.sum())})
    return pd.DataFrame(lignes)


def seuil_optimal(y_vrai, proba, **kw):
    balayage = balayer(y_vrai, proba, **kw)
    meilleur = balayage.loc[balayage["cout"].idxmin()]
    return float(meilleur["seuil"]), balayage


def comparer(y_test, proba_test, seuil_choisi, cout_fn=COUT_FN, cout_fp=COUT_FP):
    """Compare, sur le jeu de test, le seuil par défaut et le seuil retenu."""
    lignes = []
    for libelle, s in [("Seuil par défaut (0,50)", 0.50),
                       (f"Seuil optimisé ({seuil_choisi:.2f})", seuil_choisi)]:
        y_pred = (proba_test >= s).astype(int)
        cout, fn, fp = cout_total(y_test, y_pred, cout_fn, cout_fp)
        vp = int(((y_test == 1) & (y_pred == 1)).sum())
        lignes.append({
            "strategie": libelle,
            "seuil": round(s, 3),
            "clients_contactes": int(y_pred.sum()),
            "departs_evites": vp,
            "faux_negatifs": fn,
            "faux_positifs": fp,
            "cout_euros": int(cout),
        })

    # Repère supplémentaire : ne rien faire du tout.
    cout_rien, fn_rien, _ = cout_total(y_test, np.zeros_like(y_test))
    lignes.insert(0, {
        "strategie": "Ne rien faire",
        "seuil": None, "clients_contactes": 0, "departs_evites": 0,
        "faux_negatifs": fn_rien, "faux_positifs": 0, "cout_euros": int(cout_rien),
    })

    df = pd.DataFrame(lignes)
    reference = df.loc[df["strategie"].str.startswith("Seuil par défaut"), "cout_euros"].iloc[0]
    df["gain_vs_defaut_%"] = ((reference - df["cout_euros"]) / reference * 100).round(1)
    return df


def sensibilite(y_valid, proba_valid, y_test, proba_test, ratios=(2, 5, 10, 20, 50)):
    """
    Le seuil optimal dépend entièrement du rapport entre les deux coûts.

    C'est la principale fragilité de la méthode et il vaut mieux la regarder en
    face : les 500 € et 50 € sont des hypothèses, pas des mesures. Si le vrai
    rapport est de 2 pour 1 et non de 10 pour 1, le seuil optimal remonte et la
    campagne devient beaucoup plus ciblée.

    Ce tableau montre comment la recommandation se déplace selon l'hypothèse
    retenue. Il indique aussi à partir de quel moment la réponse devient
    triviale : au-delà d'un certain rapport, le modèle conseille simplement de
    contacter presque tout le monde, et n'apporte alors plus grand-chose.
    """
    lignes = []
    for r in ratios:
        cfn, cfp = float(r) * 50.0, 50.0
        s, _ = seuil_optimal(y_valid, proba_valid, cout_fn=cfn, cout_fp=cfp)
        y_pred = (proba_test >= s).astype(int)
        cout, fn, fp = cout_total(y_test, y_pred, cfn, cfp)
        cout_defaut, _, _ = cout_total(y_test, (proba_test >= 0.5).astype(int), cfn, cfp)
        lignes.append({
            "rapport_couts": f"{r}:1",
            "seuil_optimal": round(s, 3),
            "part_clients_contactes_%": round(100 * y_pred.mean(), 1),
            "departs_manques": fn,
            "gain_vs_defaut_%": round((cout_defaut - cout) / cout_defaut * 100, 1),
        })
    return pd.DataFrame(lignes)
