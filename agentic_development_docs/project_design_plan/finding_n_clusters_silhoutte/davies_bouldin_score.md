davies_bouldin_score
sklearn.metrics.davies_bouldin_score(X, labels)[source]
Compute the Davies-Bouldin score.

The score is defined as the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances. Thus, clusters which are farther apart and less dispersed will result in a better score.

The minimum score is zero, with lower values indicating better clustering.

Parameters:
X
array-like of shape (n_samples, n_features)
A list of n_features-dimensional data points. Each row corresponds to a single data point.

labels
array-like of shape (n_samples,)
Predicted labels for each sample.

Returns:
score: float
The resulting Davies-Bouldin score.

Sample code:
from sklearn.metrics import davies_bouldin_score
X = [[0, 1], [1, 1], [3, 4]]
labels = [0, 0, 1]
davies_bouldin_score(X, labels)