"""
Modèles et prétraitement.

Point central : le prétraitement est *à l'intérieur* du pipeline scikit-learn,
pas appliqué en amont sur tout le jeu de données.

C'est la différence entre un résultat honnête et un résultat faussé. Si on
normalise ou si on encode avant de découper en train/test, les statistiques
utilisées (moyennes, écarts-types, modalités connues) contiennent déjà de
l'information sur le jeu de test. Le score obtenu est alors optimiste et ne se
reproduira jamais en production. En passant par un Pipeline, chaque pli de la
validation croisée réapprend son prétraitement sur ses seules données
d'entraînement.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier

ALEA = 42


def pretraitement(numeriques, categorielles):
    return ColumnTransformer([
        ("num", StandardScaler(), numeriques),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorielles),
    ])


def catalogue(numeriques, categorielles):
    """
    Quatre modèles, du plus bête au plus élaboré.

    Le DummyClassifier n'est pas là pour faire joli : avec 26,5 % de départs,
    un modèle qui répond systématiquement "ce client reste" obtient déjà 73,5 %
    de bonnes réponses. Sans ce repère, on peut se convaincre qu'un modèle à
    79 % d'exactitude est bon, alors qu'il n'apporte presque rien.
    """
    prep = lambda: pretraitement(numeriques, categorielles)

    return {
        "Classe majoritaire": Pipeline([
            ("prep", prep()),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "Régression logistique": Pipeline([
            ("prep", prep()),
            ("clf", LogisticRegression(max_iter=2000, random_state=ALEA)),
        ]),
        "Forêt aléatoire": Pipeline([
            ("prep", prep()),
            ("clf", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=5,
                random_state=ALEA, n_jobs=-1)),
        ]),
        "Gradient boosting": Pipeline([
            ("prep", prep()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06,
                max_leaf_nodes=31, random_state=ALEA)),
        ]),
    }
