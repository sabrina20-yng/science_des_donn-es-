import sys
from pathlib import Path


sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "src"
    )
)


from dynamic_clouds import (
    DynamicClouds,
    euclidean_distance,
    mean,
)


# ============================================================
# DONNEES DE TEST
# ============================================================

DATA = [

    [1.0, 1.0],
    [1.2, 0.8],
    [0.8, 1.1],

    [5.0, 5.0],
    [5.2, 4.8],
    [4.7, 5.1],

    [9.0, 1.0],
    [8.8, 1.2],
    [9.2, 0.9],
]


# ============================================================
# TEST DISTANCE
# ============================================================

def test_euclidean_distance():

    assert euclidean_distance(
        [0, 0],
        [3, 4]
    ) == 5.0


# ============================================================
# TEST MOYENNE
# ============================================================

def test_mean():

    assert mean(
        [
            [1, 2],
            [3, 4]
        ]
    ) == [
        2.0,
        3.0
    ]


# ============================================================
# TEST K-MEANS
# ============================================================

def test_point_representation_is_kmeans_case():

    model = DynamicClouds(

        k=3,

        representation="point",

        n_prototypes=3,

        random_state=42,
    )

    labels = model.fit_predict(
        DATA
    )

    assert len(
        labels
    ) == len(DATA)

    assert len(
        model.representations
    ) == 3

    for representation in (
        model.representations
    ):

        # K-means possède
        # un seul prototype.
        assert len(
            representation.prototypes
        ) == 1

    assert model.inertia >= 0


# ============================================================
# TEST PLUSIEURS PROTOTYPES
# ============================================================

def test_multiple_prototypes():

    model = DynamicClouds(

        k=3,

        representation="prototypes",

        n_prototypes=3,

        random_state=42,
    )

    model.fit(
        DATA
    )

    assert len(
        model.representations
    ) == 3

    for representation in (
        model.representations
    ):

        assert (
            1
            <= len(
                representation.prototypes
            )
            <= 3
        )


# ============================================================
# TEST AUTRES REPRESENTATIONS
# ============================================================

def test_other_representations():

    representations = [

        "axes",

        "distribution",

        "structure",
    ]

    for representation in representations:

        model = DynamicClouds(

            k=3,

            representation=representation,

            n_prototypes=3,

            random_state=42,
        )

        labels = (
            model.fit_predict(
                DATA
            )
        )

        assert len(
            labels
        ) == len(DATA)

        assert model.inertia >= 0

        assert (
            model.n_iterations > 0
        )


# ============================================================
# TEST PREDICTION
# ============================================================

def test_predict():

    model = DynamicClouds(

        k=3,

        representation="point",

        random_state=42,
    )

    model.fit(
        DATA
    )

    predictions = (
        model.predict(
            [
                [1.0, 0.9],
                [5.0, 5.1],
                [9.0, 1.0],
            ]
        )
    )

    assert len(
        predictions
    ) == 3