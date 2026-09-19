"""
Programme principal du TP.

L'utilisateur choisit la représentation
de la nuée dynamique.
"""

import csv
from pathlib import Path

from dynamic_clouds import DynamicClouds


# ============================================================
# CHEMIN DES DONNEES
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "donnees.csv"
)


# ============================================================
# REPRESENTATIONS DISPONIBLES
# ============================================================

REPRESENTATION_NAMES = {

    "1": (
        "prototypes",
        "Ensemble de points représentatifs"
    ),

    "2": (
        "axes",
        "Axes / composantes représentatives"
    ),

    "3": (
        "distribution",
        "Distribution de probabilités (approximation)"
    ),

    "4": (
        "structure",
        "Structure représentative"
    ),

    "5": (
        "point",
        "Point unique (cas particulier K-means)"
    ),
}


# ============================================================
# CHARGEMENT CSV
# ============================================================

def load_csv(path):
    """
    Charge les données depuis un fichier CSV.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file
        )

        return [
            [
                float(row["x"]),
                float(row["y"])
            ]

            for row in reader
        ]


# ============================================================
# CHOIX DE LA REPRESENTATION
# ============================================================

def choose_representation():

    print()

    print(
        "=" * 65
    )

    print(
        "       CHOIX DE LA REPRESENTATION DE LA NUEE"
    )

    print(
        "=" * 65
    )

    print(
        "\n1. Ensemble de points représentatifs"
    )

    print(
        "2. Axes / composantes représentatives"
    )

    print(
        "3. Distribution de probabilités"
    )

    print(
        "4. Structure représentative"
    )

    print(
        "5. Point unique (cas particulier K-means)"
    )

    print(
        "=" * 65
    )

    choice = input(
        "\nChoisissez une représentation [1-5] : "
    ).strip()

    if choice not in REPRESENTATION_NAMES:

        raise ValueError(
            "Choix invalide. "
            "Veuillez sélectionner une option de 1 à 5."
        )

    representation, label = (
        REPRESENTATION_NAMES[choice]
    )

    print(
        f"\nReprésentation choisie : {label}"
    )

    return representation


# ============================================================
# AFFICHAGE DES RESULTATS
# ============================================================

def display_results(model):

    print()

    print(
        "=" * 65
    )

    print(
        "                    RESULTATS"
    )

    print(
        "=" * 65
    )

    print(
        f"Représentation : "
        f"{model.representation}"
    )

    print(
        f"Nombre de classes : "
        f"{model.k}"
    )

    print(
        f"Nombre d'itérations : "
        f"{model.n_iterations}"
    )

    print(
        f"Inertie finale : "
        f"{model.inertia:.6f}"
    )

    print(
        "\nReprésentation des classes :"
    )

    for cluster in (
        model.describe()["clusters"]
    ):

        print(
            f"\nClasse "
            f"{cluster['cluster']}"
        )

        centroid = [
            round(value, 3)
            for value
            in cluster["centroid"]
        ]

        print(
            f"  Centroïde moyen : "
            f"{centroid}"
        )

        print(
            "  Éléments représentatifs :"
        )

        for prototype in (
            cluster["prototypes"]
        ):

            formatted = [
                round(value, 3)
                for value
                in prototype
            ]

            print(
                f"    {formatted}"
            )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    # Chargement des données
    data = load_csv(
        DATA_FILE
    )

    # Choix de la représentation
    representation = (
        choose_representation()
    )

    # Création du modèle
    model = DynamicClouds(

        k=3,

        representation=representation,

        n_prototypes=3,

        max_iterations=100,

        tolerance=0.0001,

        random_state=42,
    )

    # Apprentissage
    model.fit(
        data
    )

    # Résultats
    display_results(
        model
    )

    # ========================================================
    # TEST SUR DE NOUVEAUX POINTS
    # ========================================================

    new_points = [

        [1.0, 0.9],

        [5.3, 5.0],

        [8.9, 1.0],
    ]

    predictions = (
        model.predict(
            new_points
        )
    )

    print(
        "\nClassification de nouveaux points :"
    )

    for point, label in zip(
        new_points,
        predictions
    ):

        print(
            f"  {point} "
            f"-> Classe {label + 1}"
        )

    print()

    print(
        "=" * 65
    )

    print(
        "                       FIN"
    )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()