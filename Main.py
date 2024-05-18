from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestCentroid
from prediction_strength import determine_optimal_k
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#### 1. Відкрити та зчитати наданий файл з даними
data = pd.read_csv('dataset2_l4.txt', sep=',')
print(data.head(5))

#### 2. Визначити та вивести кількість записів.
print("Кількість записів у файлі: ", data.shape[0])

#### 3. Видалити атрибут Class.
data = data.drop(columns=['Class'])
print("\n", data.head(5))

#### 4. Вивести атрибути, що залишилися. 
print(data.columns)

##### 5. Використовуючи функцію KMeans бібліотеки scikit-learn, виконати розбиття набору даних 
# на кластери з випадковою початковою ініціалізацією і вивести координати центрів кластерів. 
# Оптимальну кількість кластерів визначити на основі початкового набору даних трьома різними способами:
# 1) elbow method, 2) average silhouette method, 3) prediction strength method.
# Отримані результати порівняти і пояснити, який метод дав кращий результат і чому так (на Вашу думку). 
ncl = range(2, 18)

models = [KMeans(n_clusters=i, init='random', random_state=13, n_init='auto').fit(data) for i in ncl]

# Метод ліктя
elbows = [model.inertia_ for model in models]

plt.figure(figsize=(10, 6))
plt.plot(ncl, elbows, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Метод силуетів
silhouette = [silhouette_score(data, model.labels_) for model in models]

plt.figure(figsize=(10, 6))
plt.plot(ncl, silhouette, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method for Optimal k')
plt.show()


# # Пред стренг
# optimal_k, ps_scores = determine_optimal_k(data, max_k=17)
# print(f"The optimal number of clusters according to prediction strength method is: {optimal_k}")

# plt.figure(figsize=(10, 6))
# plt.plot(range(2, 18), ps_scores, 'bx-')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Prediction Strength')
# plt.title('Prediction Strength Method for Optimal k')
# plt.axhline(y=0.8, color='r', linestyle='--')
# plt.show()

# # Printing cluster centers for the optimal k
# optimal_kmeans = KMeans(n_clusters=optimal_k, init='random', random_state=13, n_init='auto').fit(data)
# print("Cluster centers for the optimal number of clusters (prediction strength method):")
# print(optimal_kmeans.cluster_centers_)

optimal_kmeans = models[10]
print(optimal_kmeans.cluster_centers_)


#### 6. За раніш обраної кількості кластерів багаторазово проведіть кластеризацію методом 
# k-середніх, використовуючи для початкової ініціалізації метод k-means++. 
# Виберіть найкращий варіант кластеризації. Який кількісний критерій Ви обрали для відбору найкращої кластеризації? 
k = 10
rand_st = [13, 4, 8, 7, 1]

models = [KMeans(n_clusters=k, init='k-means++', random_state=rand_i, n_init='auto').fit(data) for rand_i in rand_st]
elbows = [model.inertia_ for model in models]
silhouette = [silhouette_score(data, model.labels_) for model in models]
davies_score = [davies_bouldin_score(data, model.labels_) for model in models]
calinski_score = [calinski_harabasz_score(data, model.labels_) for model in models]

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(rand_st)+1), elbows)
plt.xlabel('Model Number')
plt.ylabel('Silhouette Score')
plt.title('Elbow Method')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(rand_st)+1), silhouette)
plt.xlabel('Model Number')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(rand_st)+1), davies_score)
plt.xlabel('Model Number')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(rand_st)+1), calinski_score)
plt.xlabel('Model Number')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.show()

kmeans_plusplus = models[2]
print(kmeans_plusplus.cluster_centers_)

#### 7. Використовуючи функцію AgglomerativeClustering бібліотеки scikitlearn, виконати розбиття набору даних на кластери. 
# Кількість кластерів обрати такою ж самою, як і в попередньому методі. Вивести координати центрів кластерів.
algor_clustering = AgglomerativeClustering(n_clusters=k)
prediction = algor_clustering.fit_predict(data)

nearest_centroid = NearestCentroid()
nearest_centroid.fit(data, prediction)
print(nearest_centroid.centroids_)


#### 8. Порівняти результати двох використаних методів кластеризації.
comp_models = [kmeans_plusplus, algor_clustering]
comp_silhoette_score = [silhouette_score(data, model.labels_) for model in comp_models]
comp_davies_scroe = [davies_bouldin_score(data, model.labels_) for model in comp_models]
comp_calinski_score = [calinski_harabasz_score(data, model.labels_) for model in comp_models]

plt.figure(figsize=(8, 6))
plt.bar(['KMeans++', 'AgglomerativeClustering'], comp_silhoette_score)
plt.xlabel('Назви методів кластеризації')
plt.ylabel('Оцінка силуету')
plt.title('Метод середнього силуету')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.bar(['KMeans++', 'AgglomerativeClustering'], comp_davies_scroe)
plt.xlabel('Назви методів кластеризації')
plt.ylabel('Оцінка силуету')
plt.title('Метод середнього силуету')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.bar(['KMeans++', 'AgglomerativeClustering'], comp_calinski_score)
plt.xlabel('Назви методів кластеризації')
plt.ylabel('Оцінка силуету')
plt.title('Метод середнього силуету')
plt.grid(True)
plt.show()