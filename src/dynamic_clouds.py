"""
Implémentation pédagogique des Nuées dynamiques.

Principe :

    initialisation
          ↓
    affectation
          ↓
    mise à jour
          ↓
    convergence

Le projet permet plusieurs représentations d'une classe :

    - point : un seul prototype, cas K-means ;
    - prototypes : plusieurs points représentatifs ;
    - axes : représentation structurée autour des dimensions ;
    - distribution : représentation simplifiée de la répartition ;
    - structure : structure représentative générale.

IMPORTANT :

Les modes "axes", "distribution" et "structure" constituent ici
des représentations pédagogiques. Ils illustrent l'idée générale
d'une représentation plus riche qu'un seul centroïde.
"""


from dataclasses import dataclass
import math
import random
from typing import List, Sequence


Point = List[float]


# ============================================================
# OUTILS MATHEMATIQUES
# ============================================================

def euclidean_distance(
    point_a: Sequence[float],
    point_b: Sequence[float]
) -> float:
    """
    Calcule la distance euclidienne entre deux points.
    """

    if len(point_a) != len(point_b):
        raise ValueError(
            "Les points doivent avoir la même dimension."
        )

    return math.sqrt(
        sum(
            (a - b) ** 2
            for a, b in zip(point_a, point_b)
        )
    )


def squared_distance(
    point_a: Sequence[float],
    point_b: Sequence[float]
) -> float:
    """
    Calcule la distance euclidienne au carré.
    """

    if len(point_a) != len(point_b):
        raise ValueError(
            "Les points doivent avoir la même dimension."
        )

    return sum(
        (a - b) ** 2
        for a, b in zip(point_a, point_b)
    )


def mean(
    points: Sequence[Sequence[float]]
) -> Point:
    """
    Calcule le centroïde d'un ensemble de points.
    """

    if not points:
        raise ValueError(
            "Impossible de calculer la moyenne d'un groupe vide."
        )

    dimension = len(points[0])

    if any(
        len(point) != dimension
        for point in points
    ):
        raise ValueError(
            "Tous les points doivent avoir la même dimension."
        )

    return [
        sum(
            point[j]
            for point in points
        ) / len(points)
        for j in range(dimension)
    ]


def unique_points(
    points: Sequence[Sequence[float]]
) -> List[Point]:
    """
    Supprime les doublons d'une liste de points.
    """

    result = []
    seen = set()

    for point in points:

        key = tuple(
            round(value, 12)
            for value in point
        )

        if key not in seen:
            seen.add(key)
            result.append(list(point))

    return result


# ============================================================
# STRUCTURE D'UNE REPRESENTATION
# ============================================================

@dataclass
class ClusterRepresentation:
    """
    Représentation finale d'une classe.
    """

    cluster_id: int
    kind: str
    prototypes: List[Point]
    centroid: Point


# ============================================================
# ALGORITHME DES NUÉES DYNAMIQUES
# ============================================================

class DynamicClouds:
    """
    Algorithme général des Nuées dynamiques.

    Une classe peut être représentée par :

    - un point ;
    - plusieurs points représentatifs ;
    - des axes ;
    - une distribution ;
    - une structure.

    Le mode "point" correspond au cas particulier K-means.
    """

    VALID_REPRESENTATIONS = {
        "point",
        "prototypes",
        "axes",
        "distribution",
        "structure",
    }

    def __init__(
        self,
        k=3,
        representation="point",
        n_prototypes=3,
        max_iterations=100,
        tolerance=0.0001,
        random_state=42,
    ):

        if k <= 0:
            raise ValueError(
                "k doit être supérieur à 0."
            )

        if representation not in self.VALID_REPRESENTATIONS:
            raise ValueError(
                "Représentation invalide."
            )

        if n_prototypes <= 0:
            raise ValueError(
                "n_prototypes doit être supérieur à 0."
            )

        self.k = k

        self.representation = representation

        self.n_prototypes = n_prototypes

        self.max_iterations = max_iterations

        self.tolerance = tolerance

        self.random_state = random_state

        self.labels = []

        self.representations = []

        self.centroids = []

        self.inertia = None

        self.n_iterations = 0

    # ========================================================
    # VALIDATION DES DONNEES
    # ========================================================

    def _validate_data(self, data):

        if not data:
            raise ValueError(
                "Les données ne peuvent pas être vides."
            )

        dimension = len(data[0])

        if dimension == 0:
            raise ValueError(
                "Les données doivent posséder au moins "
                "une dimension."
            )

        for point in data:

            if len(point) != dimension:
                raise ValueError(
                    "Tous les individus doivent avoir "
                    "la même dimension."
                )

    # ========================================================
    # INITIALISATION
    # ========================================================

    def _initialize_representations(self, data):

        rng = random.Random(
            self.random_state
        )

        # K-means :
        # un seul représentant par classe.
        if self.representation == "point":
            count = 1

        else:
            count = min(
                self.n_prototypes,
                len(data)
            )

        total_required = self.k * count

        if len(data) >= total_required:

            indices = rng.sample(
                range(len(data)),
                total_required
            )

        else:

            indices = rng.choices(
                range(len(data)),
                k=total_required
            )

        representations = []

        position = 0

        for _ in range(self.k):

            cluster_prototypes = []

            for _ in range(count):

                cluster_prototypes.append(
                    list(
                        data[
                            indices[position]
                        ]
                    )
                )

                position += 1

            representations.append(
                cluster_prototypes
            )

        return representations

    # ========================================================
    # DISTANCE ENTRE UN INDIVIDU ET UNE NUEE
    # ========================================================

    def _cloud_distance(
        self,
        point,
        prototypes
    ):
        """
        Distance entre un individu et une nuée.

        On utilise la distance vers le prototype
        le plus proche.

        Avec un seul prototype, on retrouve
        le principe de K-means.
        """

        return min(
            euclidean_distance(
                point,
                prototype
            )
            for prototype in prototypes
        )

    # ========================================================
    # AFFECTATION
    # ========================================================

    def _assign_clusters(
        self,
        data,
        representations
    ):
        """
        Affecte chaque individu à la nuée
        la plus proche.
        """

        labels = []

        for point in data:

            distances = [
                self._cloud_distance(
                    point,
                    prototypes
                )
                for prototypes
                in representations
            ]

            closest_cluster = distances.index(
                min(distances)
            )

            labels.append(
                closest_cluster
            )

        return labels

    # ========================================================
    # SELECTION DE PLUSIEURS PROTOTYPES
    # ========================================================

    def _select_representative_points(
        self,
        points
    ):
        """
        Sélectionne plusieurs points représentatifs.

        Le premier est proche du centre.

        Les suivants sont choisis progressivement
        parmi les points les plus éloignés des
        représentants déjà sélectionnés.
        """

        count = min(
            self.n_prototypes,
            len(points)
        )

        if count == len(points):

            return [
                list(point)
                for point in points
            ]

        center = mean(points)

        first = min(
            points,
            key=lambda point:
            euclidean_distance(
                point,
                center
            )
        )

        selected = [
            list(first)
        ]

        remaining = [
            list(point)
            for point in points
            if list(point) != list(first)
        ]

        while (
            len(selected) < count
            and remaining
        ):

            candidate = max(
                remaining,
                key=lambda point:
                min(
                    euclidean_distance(
                        point,
                        prototype
                    )
                    for prototype
                    in selected
                )
            )

            selected.append(
                candidate
            )

            remaining.remove(
                candidate
            )

        return selected

    # ========================================================
    # REPRESENTATION PAR AXES
    # ========================================================

    def _build_axis_representation(
        self,
        points
    ):
        """
        Représentation pédagogique par axes.

        Elle contient :

        - le centre ;
        - les points minimums ;
        - les points maximums
          de chaque dimension.
        """

        center = mean(points)

        result = [
            center
        ]

        dimension = len(center)

        for j in range(dimension):

            minimum = min(
                points,
                key=lambda point:
                point[j]
            )

            maximum = max(
                points,
                key=lambda point:
                point[j]
            )

            result.append(
                list(minimum)
            )

            result.append(
                list(maximum)
            )

        return unique_points(
            result
        )

    # ========================================================
    # REPRESENTATION PAR DISTRIBUTION
    # ========================================================

    def _build_distribution_representation(
        self,
        points
    ):
        """
        Représentation pédagogique de la distribution.

        On conserve :

        - le centre ;
        - des représentants proches
          des quartiles inférieur et supérieur.
        """

        center = mean(points)

        result = [
            center
        ]

        dimension = len(center)

        for j in range(dimension):

            ordered = sorted(
                points,
                key=lambda point:
                point[j]
            )

            q1_index = (
                len(ordered) // 4
            )

            q3_index = (
                3 * len(ordered) // 4
            )

            result.append(
                list(
                    ordered[q1_index]
                )
            )

            result.append(
                list(
                    ordered[q3_index]
                )
            )

        return unique_points(
            result
        )

    # ========================================================
    # REPRESENTATION STRUCTURELLE
    # ========================================================

    def _build_structure_representation(
        self,
        points
    ):
        """
        Représentation structurée :

        - centre ;
        - valeurs extrêmes ;
        - plusieurs éléments
          décrivant la classe.
        """

        center = mean(points)

        result = [
            center
        ]

        dimension = len(center)

        for j in range(dimension):

            minimum = min(
                points,
                key=lambda point:
                point[j]
            )

            maximum = max(
                points,
                key=lambda point:
                point[j]
            )

            result.append(
                list(minimum)
            )

            result.append(
                list(maximum)
            )

        return unique_points(
            result
        )

    # ========================================================
    # MISE A JOUR DES REPRESENTATIONS
    # ========================================================

    def _update_representations(
        self,
        data,
        labels,
        old_representations
    ):
        """
        Met à jour la représentation
        de chaque classe.
        """

        new_representations = []

        for cluster_id in range(
            self.k
        ):

            cluster_points = [
                point
                for point, label
                in zip(data, labels)
                if label == cluster_id
            ]

            # Si une classe est vide,
            # on conserve sa représentation.
            if not cluster_points:

                new_representations.append(
                    [
                        list(point)
                        for point
                        in old_representations[
                            cluster_id
                        ]
                    ]
                )

                continue

            # ------------------------------------------------
            # CAS K-MEANS
            # ------------------------------------------------

            if self.representation == "point":

                new_representation = [
                    mean(
                        cluster_points
                    )
                ]

            # ------------------------------------------------
            # PLUSIEURS PROTOTYPES
            # ------------------------------------------------

            elif (
                self.representation
                == "prototypes"
            ):

                new_representation = (
                    self._select_representative_points(
                        cluster_points
                    )
                )

            # ------------------------------------------------
            # AXES
            # ------------------------------------------------

            elif (
                self.representation
                == "axes"
            ):

                new_representation = (
                    self._build_axis_representation(
                        cluster_points
                    )
                )

            # ------------------------------------------------
            # DISTRIBUTION
            # ------------------------------------------------

            elif (
                self.representation
                == "distribution"
            ):

                new_representation = (
                    self._build_distribution_representation(
                        cluster_points
                    )
                )

            # ------------------------------------------------
            # STRUCTURE
            # ------------------------------------------------

            else:

                new_representation = (
                    self._build_structure_representation(
                        cluster_points
                    )
                )

            new_representations.append(
                new_representation
            )

        return new_representations

    # ========================================================
    # CONVERGENCE
    # ========================================================

    def _has_converged(
        self,
        old_representations,
        new_representations
    ):
        """
        Vérifie si les représentations
        sont stabilisées.

        Pour comparer deux nuées,
        on compare leurs centres moyens.
        """

        movements = []

        for old, new in zip(
            old_representations,
            new_representations
        ):

            old_center = mean(old)

            new_center = mean(new)

            movements.append(
                euclidean_distance(
                    old_center,
                    new_center
                )
            )

        maximum_movement = max(
            movements,
            default=0.0
        )

        return (
            maximum_movement
            <= self.tolerance
        )

    # ========================================================
    # INERTIE
    # ========================================================

    def _compute_inertia(
        self,
        data,
        labels,
        representations
    ):
        """
        Mesure la compacité des classes.

        Chaque individu contribue avec le carré
        de sa distance au représentant
        le plus proche de sa classe.
        """

        total = 0.0

        for point, label in zip(
            data,
            labels
        ):

            distance = (
                self._cloud_distance(
                    point,
                    representations[label]
                )
            )

            total += distance ** 2

        return total

    # ========================================================
    # APPRENTISSAGE
    # ========================================================

    def fit(self, data):
        """
        Exécute l'algorithme des Nuées dynamiques.
        """

        self._validate_data(
            data
        )

        if len(data) < self.k:

            raise ValueError(
                "Le nombre d'individus doit "
                "être supérieur ou égal à k."
            )

        representations = (
            self._initialize_representations(
                data
            )
        )

        for iteration in range(
            1,
            self.max_iterations + 1
        ):

            # 1. Affectation
            labels = (
                self._assign_clusters(
                    data,
                    representations
                )
            )

            # 2. Mise à jour
            new_representations = (
                self._update_representations(
                    data,
                    labels,
                    representations
                )
            )

            # 3. Convergence
            if self._has_converged(
                representations,
                new_representations
            ):

                representations = (
                    new_representations
                )

                self.n_iterations = (
                    iteration
                )

                break

            representations = (
                new_representations
            )

        else:

            self.n_iterations = (
                self.max_iterations
            )

        # Affectation finale
        labels = self._assign_clusters(
            data,
            representations
        )

        self.labels = labels

        self.representations = [
            ClusterRepresentation(
                cluster_id=index + 1,
                kind=self.representation,
                prototypes=prototypes,
                centroid=mean(
                    prototypes
                )
            )
            for index, prototypes
            in enumerate(
                representations
            )
        ]

        self.centroids = [
            representation.centroid
            for representation
            in self.representations
        ]

        self.inertia = (
            self._compute_inertia(
                data,
                labels,
                representations
            )
        )

        return self

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(self, data):
        """
        Classe de nouveaux individus.
        """

        if not self.representations:

            raise RuntimeError(
                "Le modèle doit être entraîné "
                "avant predict()."
            )

        representations = [
            representation.prototypes
            for representation
            in self.representations
        ]

        return self._assign_clusters(
            data,
            representations
        )

    # ========================================================
    # FIT + PREDICT
    # ========================================================

    def fit_predict(self, data):

        self.fit(data)

        return self.labels

    # ========================================================
    # DESCRIPTION DES RESULTATS
    # ========================================================

    def describe(self):

        return {
            "representation":
                self.representation,

            "k":
                self.k,

            "n_prototypes":
                self.n_prototypes,

            "iterations":
                self.n_iterations,

            "inertia":
                self.inertia,

            "clusters": [
                {
                    "cluster":
                        representation.cluster_id,

                    "centroid":
                        representation.centroid,

                    "prototypes":
                        representation.prototypes,
                }

                for representation
                in self.representations
            ],
        }
    