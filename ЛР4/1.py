import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1️
# генерация выборки
M = 1   # математическое ожидание
s = 1   # стандартное отклонение

# создаем массив из 1000 нормально распределенных значений
arr = np.random.normal(M, s, 1000)

# преобразуем в series
series = pd.Series(arr)


# 2️
# доля значений в диапазоне (M - s; M + s) - считаем среднее
within_1sigma = ((series > (M - s)) & (series < (M + s))).mean()
print(f"Доля в диапазоне (M - s; M + s): {within_1sigma:.3f}")


# 3️
# доля значений в диапазоне (M - 3s; M + 3s) - считаем среднее
within_3sigma = ((series > (M - 3 * s)) & (series < (M + 3 * s))).mean()
print(f"Доля в диапазоне (M - 3s; M + 3s): {within_3sigma:.3f}")


# 4️
# замена каждого x в серии на sqrt(x)
root_series = pd.Series(np.sqrt(series))


# 5️
# среднее арифметическое после sqrt
mean_root = root_series.mean(skipna=True)
print(f"Среднее значение для получившихся значений: {mean_root:.3f}")


# 6️
# создание DataFrame
df = pd.DataFrame({
    "number": series, # столбцы
    "root": root_series
})

print("\nПервые 6 строк DataFrame:")
print(df.head(6))


# 7️
# поиск записей, где sqrt(x) ∈ [1.8, 1.9]
result = df.query("root >= 1.8 and root <= 1.9")
print("\nЗаписи, где root ∈ [1.8; 1.9]:")
print(result)



# построение графика для 4

# гистограммы
plt.figure(figsize=(10, 5))

# исходные значения
plt.hist(series, bins=30, alpha=0.6, label='Исходные значения')

# после sqrt() (убираем NaN)
plt.hist(root_series.dropna(), bins=30, alpha=0.6, label='Получившиеся значения (после sqrt(x))')

# оформление графика
plt.title("Сравнение распределений: исходных и получившихся значений после sqrt(x)")
plt.xlabel("Значение")
plt.ylabel("Частота")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()