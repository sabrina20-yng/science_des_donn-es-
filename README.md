# science_des_donn-es-
# TP — Implémentation des Nuées dynamiques et K-means from scratch

## 1. Présentation

Ce projet consiste à implémenter un algorithme de classification
non supervisée inspiré du principe des Nuées dynamiques.

Les Nuées dynamiques, proposées par Edwin Diday en 1971,
peuvent être considérées comme une généralisation de la méthode
des k-means.

Les deux approches reposent sur un principe itératif similaire :

1. initialisation des classes ;
2. affectation des individus selon un critère de proximité ;
3. mise à jour de la représentation des classes ;
4. répétition jusqu'à stabilisation.

La différence fondamentale concerne la représentation des classes.

Dans K-means, chaque classe est représentée par un seul
centre de gravité ou centroïde.

Dans les Nuées dynamiques, une classe peut être représentée
par une nuée composée de plusieurs éléments représentatifs.

---

## 2. K-means comme cas particulier

Dans notre projet, K-means correspond au cas où :

```text
une classe
     ↓
un seul prototype
     ↓
centroïde