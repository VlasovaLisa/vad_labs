import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error

# 1
# читаем данные из boston.csv
df = pd.read_csv('boston.csv')
print("Первые строки набора данных:\n")
print(df.head())  # вывод первых 5 строк


# 2
# проверка, что все признаки числового типа
print("\n -- 2. Типы данных по столбцам: --")
print(df.dtypes)


# 3 
# проверка пропусков и заполнение медианой 
print("\n -- 3. Количество пропусков в каждом столбце: --")
print(df.isna().sum())
df = df.fillna(df.median(numeric_only=True))


# 4
# вычисление корреляционной матрицы
print("\n -- 4. Корреляционная матрица: --")
corr = df.corr(numeric_only=True) # для всех пар
print(corr)


# 5 
# тепловая карта корреляций 
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("5. Тепловая карта по корреляционной матрице")
plt.show()


# 6
# выбор признаков с наибольшей корреляцией с MEDV 
top6 = corr['MEDV'].abs().sort_values(ascending=False).head(7)
print("\n -- 6. Топ-6 признаков по корреляции с MEDV: --")
print(top6)
top6 = top6.index.drop('MEDV').tolist()


# 7
# диаграммы рассеяния для выбранных признаков
for f in top6:
    sns.scatterplot(x=df[f], y=df['MEDV'])
    plt.title(f"7. {f} vs MEDV (corr={corr.loc[f,'MEDV']:.2f})")
    plt.xlabel(f)
    plt.ylabel("MEDV")
    plt.show()


# 8
# визуальная проверка связи 
# в данном случае оставляем все выбранные


# 9
# формируем набор признаков (X) и целевую переменную (y)
X = df[top6] # матрица признаков
y = df['MEDV'] # вектор целевой переменной


# 10
# разделение данных на обучающую и тестовую выборки (80/20)
X_train, X_test, y_train, y_test = train_test_split(X,  y, test_size=0.2, random_state=42, shuffle=True)


# 11
# обучение линейной регрессии
lin = LinearRegression() # создаем объект модели линейной регрессии
lin.fit(X_train, y_train) # обучаем модель


# 12
# прогнозирование на обучающей и тестовой выборках 
y_pred_train = lin.predict(X_train) # для обучающего набора
y_pred_test = lin.predict(X_test) # для тестового набора



# 13
# вычисление метрик R^2 и RMSE
r2_train = r2_score(y_train, y_pred_train) # коэффициент детерминации (насколько хорошо модель объясняет данные)
r2_test = r2_score(y_test, y_pred_test)
rmse_train = mean_squared_error(y_train, y_pred_train)**0.5 # корень из среднеквадратичной ошибки (на сколько модель ошибается)
rmse_test = mean_squared_error(y_test, y_pred_test)**0.5

# выводим результаты
print("\n -- 13. Линейная регрессия: --")
print(f"Train R^2={r2_train:.3f}, RMSE={rmse_train:.3f}")
print(f"Test  R^2={r2_test:.3f}, RMSE={rmse_test:.3f}")



# 14
# Boxplot для целевого признака MEDV (поиск выбросов) 
sns.boxplot(x=df['MEDV'])
plt.title("14. Boxplot для MEDV")
plt.show()

# считаем выбросами MEDV вне [5.5; 36.5]
df = df[df['MEDV'] < 36.5]
df = df[df['MEDV'] > 5.5]



# 15
# повторное обучение после удаления выбросов 
X = df[top6] # заново формируем X и y
y = df['MEDV'] # целевой признак 
# разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
# test_size=0.2 -- 20% для тестирования
# random_state=42 -- случайное разбиение
# shuffle=True -- перемешивание данных перед разбиением (чтобы не было смещений)

lin.fit(X_train, y_train) # переобучаем линейную модель

# делаем новые прогнозы
y_pred_train = lin.predict(X_train) # прогноз модели на тренировочных данных
y_pred_test = lin.predict(X_test) # прогноз модели на тестовых данных
# оценка качества модели
r2_train_clean = r2_score(y_train, y_pred_train) # коэф детерминации
r2_test_clean = r2_score(y_test, y_pred_test)
rmse_train_clean = mean_squared_error(y_train, y_pred_train) ** 0.5 # среднеквадратичная ошибка
rmse_test_clean = mean_squared_error(y_test, y_pred_test) ** 0.5

print("\n -- 15. После удаления выбросов: --")
print(f"Train R^2 = {r2_train_clean:.3f}, RMSE = {rmse_train_clean:.3f}")
print(f"Test  R^2 = {r2_test_clean:.3f}, RMSE = {rmse_test_clean:.3f}")

# сравнение с результатами до очистки
# R² (коэффициент детерминации)
# показывает насколько хорошо модель объясняет дисперсию данных
# (доля объясненной вариации). Чем больше, тем лучше (идеально = 1).
if 'r2_test' in locals():
    if r2_test_clean > r2_test:
        print("--> Качество модели улучшилось после удаления выбросов")
    elif r2_test_clean < r2_test:
        print("--> Качество модели ухудшилось после удаления выбросов")
    else:
        print("--> Удаление выбросов не повлияло на качество модели")
        



# 16
# гребневая (Ridge) регрессия 
# линейная регрессия, но плюсом Ridge добавляет штраф (регуляризацию) за большие коэффициенты
ridge = Ridge(alpha=1.0) # создаем модель
ridge.fit(X_train, y_train) # обучаем модель

y_pred_train_ridge = ridge.predict(X_train)  # прогноз на обучающей
y_pred_test_ridge = ridge.predict(X_test)    # прогноз на тестовой

# для обучающей выборки
r2_train_ridge = r2_score(y_train, y_pred_train_ridge)
rmse_train_ridge = mean_squared_error(y_train, y_pred_train_ridge) ** 0.5

# для тестовой выборки
r2_test_ridge = r2_score(y_test, y_pred_test_ridge)
rmse_test_ridge = mean_squared_error(y_test, y_pred_test_ridge) ** 0.5

# выводим результаты
print("\n -- 16. Гребневая регрессия (Ridge) --")
print(f"Train R^2 = {r2_train_ridge:.3f}, RMSE = {rmse_train_ridge:.3f}")
print(f"Test  R^2 = {r2_test_ridge:.3f}, RMSE = {rmse_test_ridge:.3f}")

# выходит предупреждение от SciPy, что матрица признаков (X) или ее производная (XᵀX) имеет плохую численную обусловленность



# 17
# полиномиальная регрессия 
# создаем конвейер: полином 3-й степени + гребневая регрессия
poly = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    Ridge(alpha=1.0)
) 
poly.fit(X_train, y_train) # обучаем на тренировочных данных

# делаем прогноз на обеих выборках
y_pred_train_poly = poly.predict(X_train)
y_pred_test_poly = poly.predict(X_test)

# для обучающей выборки
r2_train_poly = r2_score(y_train, y_pred_train_poly)
rmse_train_poly = mean_squared_error(y_train, y_pred_train_poly) ** 0.5

# для тестовой выборки
r2_test_poly = r2_score(y_test, y_pred_test_poly)
rmse_test_poly = mean_squared_error(y_test, y_pred_test_poly) ** 0.5

# выводим результаты
print("\n -- 17. Полиномиальная регрессия --")
print(f"Train R^2 = {r2_train_poly:.3f}, RMSE = {rmse_train_poly:.3f}")
print(f"Test  R^2 = {r2_test_poly:.3f}, RMSE = {rmse_test_poly:.3f}")




# ---------------- Итоговое сравнение всех моделей --------------------

# создаем таблицу с результатами всех 3 моделей
results = pd.DataFrame({
    'Модель': ['Линейная', 'Ridge', 'Полиномиальная (3)'],
    'R^2 (train)': [r2_train_clean, r2_train_ridge, r2_train_poly],
    'R^2 (test)':  [r2_test_clean,  r2_test_ridge,  r2_test_poly],
    'RMSE (train)': [rmse_train_clean, rmse_train_ridge, rmse_train_poly],
    'RMSE (test)':  [rmse_test_clean,  rmse_test_ridge,  rmse_test_poly]
})

# выводим таблицу
print("\n --- Итоговое сравнение моделей ---")
print(results.round(3))

# R^2
# чем выше столбец, тем лучше модель объясняет вариацию данных
# (то есть точнее предсказывает цены)
plt.figure(figsize=(8, 5))
sns.barplot(data=results, x='Модель', y='R^2 (test)', hue='Модель', palette='mako', legend=False)
plt.title("Сравнение качества моделей по R^2 (тестовая выборка)")
plt.ylabel("R^2 (test)")
plt.show()

# RMSE
# чем ниже столбец, тем меньше средняя ошибка предсказания,
# то есть модель делает более точные прогнозы
plt.figure(figsize=(8, 5))
sns.barplot(data=results, x='Модель', y='RMSE (test)', hue='Модель', palette='flare', legend=False)
plt.title("Сравнение качества моделей по RMSE (тестовая выборка)")
plt.ylabel("RMSE (test)")
plt.show()
