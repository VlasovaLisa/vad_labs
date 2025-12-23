import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# функция поиска точки в файле
def find_star(file, x, y):
    data = pd.read_csv(file, sep=';', decimal=',') # читаем файл с указанием, что десятичный разделитель - запятая
    match = data[(data['X'] == x) & (data['Y'] == y)] # фильтрация: выбираем строки где X и Y точно равны x и y
    # проверка и вывод результата
    if not match.empty:
        print("Точка найдена:")
        print(match)
        return match
    else:
        print("Точка не найдена")
        return None



# чтение данных
df = pd.read_csv("27_B_17834.csv", encoding='utf-8', delimiter=';')

print(df.head()) # вывод первых 5 строк для примера

# исправляем запятые
df['X'] = df['X'].str.replace(',', '.').astype(float)
df['Y'] = df['Y'].str.replace(',', '.').astype(float)

# извлекаем массив координат
data = df[['X', 'Y']].values


# 1
# кластеризация KMeans
# создаем экземпляр KMeans на 3 кластера
# random_state задает зерно генератора случайных чисел, чтобы результаты были воспроизводимы
kmeans = KMeans(n_clusters=3, random_state=0)

# fit_predict одновременно обучает модель на данных и возвращает метки кластеров для каждой точки
# результат сохраняем в колонку 'cluster' датафрейма df, чтобы привязывать точки к кластерам
df['cluster'] = kmeans.fit_predict(data)



# 2
# поиск центроидов 

# список, в котором для каждого кластера будет храниться координата центроида (точка из набора, минимизирующая сумму расстояний)
real_centroids = []

# sorted(df['cluster'].unique()) — перебираем метки кластеров в упорядоченном виде
for cluster_id in sorted(df['cluster'].unique()):

    # выбираем все точки, принадлежащие данному кластеру, и берем только столбцы X и Y
    # .values дает массив точек в этом кластере
    cluster_points = df[df['cluster'] == cluster_id][['X', 'Y']].values

    # построение матрицы попарных евклидовых расстояний между точками кластера
    # distances[i, j] = расстояние между i-й и j-й точками
    distances = cdist(cluster_points, cluster_points, 'euclidean')
    
    # для каждой точки i вычисляем сумму расстояний до всех остальных: sums[i]
    # в матрице расстояний ищем сумму по каждой строке, чтобы найти минимум
    sums = distances.sum(axis=1)

    # np.argmin(sums) возвращает индекс точки с минимальной суммой расстояний
    # centroid — координаты этой точки (внутри кластера)
    centroid = cluster_points[np.argmin(sums)]

    # добавляем найденную точку в список центроидов
    real_centroids.append(centroid)

# преобразуем список в массив формы (k, 2), где k — число кластеров, и координаты центров
real_centroids = np.array(real_centroids)



# 3
# фильтрация точек в заданном радиусе R=3

def filter_points_within_radius(data, labels, cluster_id, radius=3):

    # берем все точки, у которых метка == cluster_id
    cluster_points = data[labels == cluster_id]

    # в этом методе центр окружности берется как среднее арифметическое точек кластера (не центроид для кластера)
    # .mean(axis=0) - вычисляет среднее значение по каждому столбцу в массиве
    centroid = cluster_points.mean(axis=0)

    # distances — евклидовы расстояния от каждой точки к вычисленному среднему центроиду
    distances = np.linalg.norm(cluster_points - centroid, axis=1)

    # возвращаем только те точки, у которых расстояние <= radius
    return cluster_points[distances <= radius]

# применяем фильтрацию ко всем кластерам и собираем результаты
filtered_data = []
labels = df['cluster'].values

# перебираем кластеры у порядоченном виде
for cluster_id in sorted(df['cluster'].unique()):
    # добавляем массив отфильтрованных точек для текущего кластера в список
    filtered_data.append(filter_points_within_radius(data, labels, cluster_id))

# np.vstack объединяет список массивов в один массив (сложение по вертикали)
filtered_data = np.vstack(filtered_data)

# превращаем результат фильтрации в DataFrame для удобства отображения
filtered_df = pd.DataFrame(filtered_data, columns=["X", "Y"])



# 4
# графики (центроиды + данные)
plt.figure(figsize=(7, 7))

# рисуем все точки
plt.scatter(df['X'], df['Y'], c=df['cluster'], cmap='viridis', s=10, alpha=0.6)

# рисуем центроиды
plt.scatter(real_centroids[:, 0], real_centroids[:, 1],
            c='black', s=80, marker='o', label='Центроиды')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Кластеризация с центроидами")
plt.legend()
plt.grid(True)
plt.show()



# 5
# вывод координат центроидов и поиск в файле

# для найденных центроидов печатаем их координаты
# и вызываем find_star для проверки, есть ли такая запись в датасете
for i, c in enumerate(real_centroids):
    print(f"\nКластер {i+1}: центроид ({c[0]:.3f}, {c[1]:.3f})")
    find_star("27_B_17834.csv", c[0], c[1])
