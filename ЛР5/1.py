import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1
# загрузка данных 
df = pd.read_csv(
    "weather1.csv",
    sep=";",
    quotechar='"',
    encoding="utf-8",
    usecols=["Местное время в Перми", "T", "P", "U", "Ff", "N", "H", "VV"]
)

# преобразуем столбец времени в datetime
df["Местное время в Перми"] = pd.to_datetime(df["Местное время в Перми"], errors="coerce")

# удаляем строки с пропущенными значениями
df = df.dropna(subset=["T", "U", "N"])



# 2 
# диаграмма зависимости температуры от влажности с Matplotlib
plt.figure(figsize=(8, 5))
plt.scatter(df["T"], df["U"], color="dodgerblue", alpha=0.6, edgecolors="k")
plt.title("2.Зависимость температуры от влажности (Matplotlib)", fontsize=14)
plt.xlabel("Температура (°C)", fontsize=12)
plt.ylabel("Относительная влажность (%)", fontsize=12)
plt.grid(True)
plt.show()

# диаграмма зависимости температуры от влажности с Seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="T", y="U", hue="U", palette="coolwarm", edgecolor="none")
plt.title("2.Зависимость температуры от влажности (Seaborn)", fontsize=14)
plt.xlabel("Температура (°C)", fontsize=12)
plt.ylabel("Относительная влажность (%)", fontsize=12)
plt.show()



# 3
# диаграмма с выделением точек по облачности с Matplotlib

# облачность (N) может содержать строки вида "100%." — извлекаем числовые значения
df["N_clean"] = df["N"].astype(str).str.extract(r"(\d+)")   # достаем цифры
df["N_clean"] = pd.to_numeric(df["N_clean"], errors="coerce")  # преобразуем в числа

cloud_100 = df[df["N_clean"] == 100]
cloud_other = df[df["N_clean"] != 100]

plt.figure(figsize=(8, 5))
plt.scatter(cloud_other["T"], cloud_other["U"], color="red", alpha=0.6, label="Облачность ≠ 100%")
plt.scatter(cloud_100["T"], cloud_100["U"], color="blue", alpha=0.6, label="Облачность = 100%")

plt.title("3.Температура и влажность (окраска по облачности)", fontsize=14)
plt.xlabel("Температура (°C)", fontsize=12)
plt.ylabel("Относительная влажность (%)", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()



# 4
# линейная диаграмма изменения температуры во времени с Matplotlib
plt.figure(figsize=(10, 5))
plt.plot(df["Местное время в Перми"], df["T"], color="green", linewidth=1.5)
plt.title("4.Изменение температуры во времени (Пермь)", fontsize=14)
plt.xlabel("Местное время", fontsize=12)
plt.ylabel("Температура (°C)", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()



# 5
# среднемесячная температура и столбчатая диаграмма
# добавляем столбец "Месяц"
df["Месяц"] = df["Местное время в Перми"].dt.month

# группируем и считаем среднюю температуру по каждому месяцу
monthly_avg = df.groupby("Месяц")["T"].mean().reset_index()

# диаграмма Matplotlib
plt.figure(figsize=(8, 5))
plt.bar(monthly_avg["Месяц"], monthly_avg["T"], color="skyblue", edgecolor="k")
plt.title("5.Среднемесячная температура в Перми (Matplotlib)", fontsize=14)
plt.xlabel("Месяц")
plt.ylabel("Средняя температура (°C)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# альтернатива Seaborn
plt.figure(figsize=(8, 5))
sns.barplot(data=monthly_avg, x="Месяц", y="T", palette="coolwarm")
plt.title("5.Среднемесячная температура в Перми (Seaborn)", fontsize=14)
plt.xlabel("Месяц")
plt.ylabel("Средняя температура (°C)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()



# 6
# ленточная (горизонтальная) диаграмма по облачности
# считаем количество наблюдений для каждого варианта облачности
cloud_counts = df["N"].value_counts().reset_index()
cloud_counts.columns = ["Облачность", "Количество"]

# диаграмма Matplotlib
plt.figure(figsize=(10, 6))
plt.barh(cloud_counts["Облачность"], cloud_counts["Количество"], color="cornflowerblue")
plt.title("6.Количество наблюдений по облачности", fontsize=14)
plt.xlabel("Количество наблюдений")
plt.ylabel("Вариант облачности")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()



# 7
# гистограмма частот температуры Matplotlib
plt.figure(figsize=(8, 5))
plt.hist(df["T"], bins=10, color="orange", edgecolor="black", alpha=0.7)
plt.title("7.Гистограмма частот температуры", fontsize=14)
plt.xlabel("Температура (°C)")
plt.ylabel("Количество наблюдений (частота)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()



# 8
# Boxplot атмосферного давления по группам видимости - ящик с усами
# создаем группы по VV
def visibility_group(v):
    if v < 5:
        return "менее 5 км"
    elif 5 <= v <= 15:
        return "5–15 км"
    else:
        return "более 15 км"

df["Группа видимости"] = df["VV"].apply(visibility_group)

# проверка распеделения по группам
print(df["Группа видимости"].value_counts())

# строим boxplot - показывает медиану, квартиль и выбросы атмосферного давления (P) в каждой группе видимости
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Группа видимости", y="P", palette="Set2")
plt.title("8.Распределение атмосферного давления по группам видимости", fontsize=14)
plt.xlabel("Группа видимости")
plt.ylabel("Атмосферное давление (мм. рт. ст.)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()



# 9
# круговая диаграмма по высоте основания облаков (H)

# группируем данные
h_counts = df["H"].value_counts().reset_index()
h_counts.columns = ["Высота основания облаков (м)", "Количество"]

# сортируем по высоте (для логичного порядка)
h_counts = h_counts.sort_values(by="Высота основания облаков (м)")

# построение красивой диаграммы Matplotlib
plt.figure(figsize=(9, 7))
wedges, texts, autotexts = plt.pie(
    h_counts["Количество"],
    autopct="%1.1f%%",
    startangle=120,
    colors=plt.cm.Set3.colors,
    wedgeprops={"edgecolor": "white"},
    pctdistance=0.8
)

# добавляем легенду справа, чтобы не перекрывала диаграмму
plt.legend(
    wedges,
    h_counts["Высота основания облаков (м)"],
    title="Высота основания облаков (м)",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=10
)

plt.title("9.Распределение по высоте основания облаков", fontsize=14)
plt.tight_layout()
plt.show()



# построение диаграммы стандартной
plt.figure(figsize=(7, 7))
plt.pie(
    h_counts["Количество"],
    labels=h_counts["Высота основания облаков (м)"],
    autopct="%1.1f%%",
    startangle=90,
    colors=plt.cm.Paired.colors
)
plt.title("9.Распределение по высоте основания облаков", fontsize=14)
plt.tight_layout()
plt.show()
