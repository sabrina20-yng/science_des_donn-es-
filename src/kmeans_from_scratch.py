"""
Implémentation de K-means from scratch.

Dans ce projet, K-means est considéré comme
le cas particulier des Nuées dynamiques où
chaque classe possède un seul représentant :
son centroïde.
"""

from dynamic_clouds import (
    DynamicClouds,
    euclidean_distance,
    mean,
)


class KMeansFromScratch(DynamicClouds):
    """
    Version spécialisée des Nuées dynamiques.

    Une seule représentation est utilisée
    pour chaque classe.
    """

    def __init__(
        self,
        k=3,
        max_iterations=100,
        tolerance=0.0001,
        random_state=42,
    ):

        super().__init__(
            k=k,
            representation="point",
            n_prototypes=1,
            max_iterations=max_iterations,
            tolerance=tolerance,
            random_state=random_state,
        )


__all__ = [
    "KMeansFromScratch",
    "euclidean_distance",
    "mean",
]