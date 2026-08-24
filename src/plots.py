"""Figures du projet."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

BLEU, ROUGE, GRIS, ORANGE = "#2a7ab9", "#c0392b", "#7f8c8d", "#e67e22"


def _propre(ax):
    ax.grid(alpha=.25, linestyle=":")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def profil_risque(df, chemin):
    """Trois lectures du départ : contrat, ancienneté, nombre de services."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    t = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
    axes[0].bar(range(len(t)), t.values * 100, color=[ROUGE, ORANGE, BLEU])
    axes[0].set_xticks(range(len(t)))
    axes[0].set_xticklabels(["Mensuel", "1 an", "2 ans"], fontsize=9)
    axes[0].set_ylabel("Taux de départ (%)")
    axes[0].set_title("Type de contrat", weight="bold", fontsize=11)
    for i, v in enumerate(t.values * 100):
        axes[0].text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9, weight="bold")

    bornes = [0, 6, 12, 24, 48, 100]
    libelles = ["0-6", "7-12", "13-24", "25-48", "49+"]
    tranche = pd.cut(df["tenure"], bins=bornes, labels=libelles, include_lowest=True)
    g = df.groupby(tranche, observed=True)["Churn"].mean().reindex(libelles)
    axes[1].plot(range(len(g)), g.values * 100, marker="o", color=ROUGE, lw=2.2, ms=7)
    axes[1].set_xticks(range(len(g)))
    axes[1].set_xticklabels(libelles, fontsize=9)
    axes[1].set_xlabel("Ancienneté (mois)")
    axes[1].set_ylabel("Taux de départ (%)")
    axes[1].set_title("Ancienneté", weight="bold", fontsize=11)

    g2 = df.groupby("nb_services")["Churn"].mean()
    axes[2].bar(g2.index, g2.values * 100, color=BLEU)
    axes[2].set_xlabel("Nombre de services souscrits")
    axes[2].set_ylabel("Taux de départ (%)")
    axes[2].set_title("Équipement", weight="bold", fontsize=11)

    for a in axes:
        _propre(a)
    fig.suptitle("Qui part, et quand", weight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def courbes_roc(resultats, y_test, chemin):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    couleurs = {"Régression logistique": BLEU, "Forêt aléatoire": ORANGE,
                "Gradient boosting": ROUGE}
    for nom, proba in resultats.items():
        if nom not in couleurs:
            continue
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, lw=2.1, color=couleurs[nom],
                label=f"{nom} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=GRIS, lw=1.2, label="Hasard")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbes ROC sur le jeu de test", weight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _propre(ax)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def courbe_cout(balayage, seuil, chemin):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(balayage["seuil"], balayage["cout"] / 1000, color=BLEU, lw=2.2)

    c_opt = balayage.loc[balayage["cout"].idxmin()]
    c_def = balayage.iloc[(balayage["seuil"] - 0.5).abs().idxmin()]

    ax.scatter([c_opt["seuil"]], [c_opt["cout"] / 1000], color=ROUGE, s=110, zorder=5)
    ax.annotate(f"optimum : seuil {c_opt['seuil']:.2f}\n{c_opt['cout'] / 1000:.1f} k€",
                (c_opt["seuil"], c_opt["cout"] / 1000),
                textcoords="offset points", xytext=(12, 16), fontsize=9,
                color=ROUGE, weight="bold")
    ax.scatter([c_def["seuil"]], [c_def["cout"] / 1000], color=GRIS, s=90, zorder=5)
    ax.annotate(f"défaut : seuil 0,50\n{c_def['cout'] / 1000:.1f} k€",
                (c_def["seuil"], c_def["cout"] / 1000),
                textcoords="offset points", xytext=(12, 14), fontsize=9, color=GRIS)

    ax.set_xlabel("Seuil de décision")
    ax.set_ylabel("Coût total (k€)")
    ax.set_title("Le seuil par défaut n'est pas le seuil optimal", weight="bold")
    _propre(ax)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def importances(df, chemin):
    d = df.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(d)), d["importance"], color=BLEU,
            xerr=d["ecart_type"], error_kw={"ecolor": GRIS, "lw": 1})
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["variable"], fontsize=9)
    ax.set_xlabel("Baisse d'AUC quand la variable est mélangée")
    ax.set_title("Ce sur quoi le modèle s'appuie réellement", weight="bold")
    _propre(ax)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
