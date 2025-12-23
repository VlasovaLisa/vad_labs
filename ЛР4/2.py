import pandas as pd
import matplotlib.pyplot as plt

# 1️
# загрузка данных
df = pd.read_csv("athlete_events.csv")

print("\nПервые 3 строки файла:")
print(df.head(3))


# 2️
# информация о данных
print("\nИнформация о данных:")
print(df.info())

# количество непустых значений по каждому столбцу
print("\nКоличество непустых значений:")
print(df.count())

# количество пропущенных данных
missing = df.isna().sum()
print("\nКоличество пропусков в каждом столбце:")
print(missing)

print("\nБольше всего пропусков в:", missing.idxmax(), "(", missing.max(), "пропусков)")



# 3️
# статистическая информация по возрасту, росту и весу
print("\nСтатистика по возрасту, росту и весу:")
print(df[["Age", "Height", "Weight"]].describe())



# 4️
# ответы на вопросы


# 4.1
# самый молодой участник в 1992 году
young_1992 = df[df["Year"] == 1992]
min_age = young_1992["Age"].min()
youngest = young_1992[young_1992["Age"] == min_age][["Name", "Age", "Event"]]
print(f"\n4.1) Самый молодой участник в 1992 году:")
print(youngest)


# 4.2
# все виды спорта
sports = df["Sport"].unique()
print("\n 4.2) Все виды спорта, которые были на Олимпиадах:")
print(sorted(sports))


# 4.3 
# средний рост теннисисток
tennis_2000 = df[(df["Year"] == 2000) & (df["Sex"] == "F") & (df["Sport"] == "Tennis")]
mean_height_tennis = tennis_2000["Height"].mean()
print(f"\n4.3) Средний рост теннисисток 2000 года: {mean_height_tennis:.2f} см")


# 4.4
# сколько золотых медалей в настольном теннисе выиграл Китай в 2008 году
china_gold_2008 = df[
    (df["Year"] == 2008)
    & (df["Team"] == "China")
    & (df["Sport"] == "Table Tennis")
    & (df["Medal"] == "Gold")
]
print(f"\n4.4) Китай выиграл в настольном теннисе в 2008 году золотых медалей: {len(china_gold_2008)}")


# 4.5
# изменение количества видов спорта: 2004 vs 1988 (летние)
sports_1988 = df[(df["Year"] == 1988) & (df["Season"] == "Summer")]["Sport"].nunique()
sports_2004 = df[(df["Year"] == 2004) & (df["Season"] == "Summer")]["Sport"].nunique()
print(f"\n4.5) В 1988 году — {sports_1988} видов спорта, в 2004 — {sports_2004}.")
print(f"Изменение: {sports_2004 - sports_1988} видов.")


# 4.6
# гистограмма возраста мужчин-керлингистов на олимпиаде 2014 года
curling_2014 = df[(df["Year"] == 2014) & (df["Sport"] == "Curling") & (df["Sex"] == "M")]
plt.hist(curling_2014["Age"].dropna(), bins=10)
plt.title("Возраст мужчин-керлингистов на Олимпиаде 2014 года")
plt.xlabel("Возраст")
plt.ylabel("Количество спортсменов")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# 4.7
# зимняя олимпиада 2006 — медали и средний возраст по странам
winter_2006 = df[(df["Year"] == 2006) & (df["Season"] == "Winter")]

medals_age = (
    winter_2006.groupby("NOC") # группируем по странам
    .agg(
        medals=("Medal", lambda x: x.notna().sum()),
        avg_age=("Age", "mean")
    )
    .query("medals > 0")
    .sort_values("medals", ascending=False)
)

print("\n4.7) Количество медалей и средний возраст спортсменов:")
print(medals_age)


# 4.8
# сводная таблица: сколько медалей каждого достоинства у каждой страны (2006, зимние)
pivot_medals = (
    winter_2006.pivot_table(
        index="NOC",
        columns="Medal",
        values="ID",
        aggfunc="count",
        fill_value=0
    )
)
print("\n4.8) Сводная таблица по медалям:")
print(pivot_medals)
