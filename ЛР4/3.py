import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# загрузка данных
df = pd.read_csv("telecom_churn.csv")

# оставляем только нужные столбцы
df = df[[
    "State", "Area code", "International plan", "Number vmail messages",
    "Total day minutes", "Total day calls",
    "Total eve minutes", "Total eve calls",
    "Total night minutes", "Total night calls",
    "Customer service calls", "Churn"
]]


# 1
# информация о данных
print("Информация о данных:")
print(df.info())

# проверка наличия пропусков
print("\nКоличество пропусков в каждом столбце:")
print(df.isna().sum())



# 2️
# количество активных и потерянных клиентов
# False - активный клиент
# True - потерянный клиент
print("\nКоличество клиентов по категориям Churn:")
churn_counts = df["Churn"].value_counts()
print(churn_counts)

# в процентах
churn_percent = df["Churn"].value_counts(normalize=True) * 100
print("\nПроцентное распределение:")
print(churn_percent)



# 3️
# добавим столбец «Продолжительность одного звонка»
df["Total calls"] = df["Total day calls"] + df["Total eve calls"] + df["Total night calls"]
df["Total minutes"] = df["Total day minutes"] + df["Total eve minutes"] + df["Total night minutes"]

# избегаем деления на ноль
df["Avg call duration"] = df["Total minutes"] / df["Total calls"]

# сортировка по убыванию
print("\nТоп-10 клиентов по средней длительности звонка:")
print(df.sort_values("Avg call duration", ascending=False).head(10))



# 4️
# средняя продолжительность звонка по группам Churn
avg_duration_by_churn = df.groupby("Churn")["Avg call duration"].mean()
print("\nСредняя продолжительность одного звонка (по Churn):")
print(avg_duration_by_churn)

diff_duration = avg_duration_by_churn[True] - avg_duration_by_churn[False]
print(f"Разница между потерянными и активными клиентами: {diff_duration:.2f} мин.")



# 5️
# среднее количество звонков в поддержку по Churn
support_calls = df.groupby("Churn")["Customer service calls"].mean()
print("\nСреднее количество звонков в поддержку (по Churn):")
print(support_calls)



# 6️
# таблица сопряженности по Churn и Customer service calls
ct = pd.crosstab(df["Customer service calls"], df["Churn"], normalize='index') * 100
print("\nПроцент оттока в зависимости от количества звонков в поддержку:")
print(ct)

# при каком количестве звонков процент оттока > 40%
high_churn_calls = ct[ct[True] > 40]
print("\nКоличество звонков в поддержку, при которых отток > 40%:")
print(high_churn_calls)



# 7️
# таблица сопряженности Churn и International plan
ct_plan = pd.crosstab(df["International plan"], df["Churn"], normalize='index') * 100
print("\nПроцент оттока в зависимости от наличия международного плана:")
print(ct_plan)

# вывод
if ct_plan.loc["Yes", True] > ct_plan.loc["No", True]:
    print("\n---> Процент оттока среди клиентов с международным планом существенно выше.")
else:
    print("\n---> Процент оттока среди клиентов без международного плана выше или равен.")



# 8️
# прогнозируемый отток
# если есть международный план ИЛИ больше 3 звонков в поддержку → прогнозируем отток (True)
df["Predicted churn"] = (df["International plan"] == "Yes") | (df["Customer service calls"] > 3)

# сравнение реального и прогнозного оттока
confusion = pd.crosstab(df["Churn"], df["Predicted churn"], rownames=["Реальный"], colnames=["Прогнозируемый"])

print("\nТаблица сравнения (реальный vs прогнозируемый):")
print("\nТаблица 1: Клиенты, которые остались (Churn = False)")
print(confusion.loc[False])

print("\nТаблица 2: Клиенты, которые ушли (Churn = True)")
print(confusion.loc[True])

# подсчет ошибок
# ошибка 1 рода (ложноотрицательная): клиент ушёл, но модель сказала False
false_negative = confusion.loc[True, False]
# ошибка 2 рода (ложноположительная): клиент остался, но модель сказала True
false_positive = confusion.loc[False, True]

total_true = confusion.loc[True].sum()
total_false = confusion.loc[False].sum()

error_type1 = false_negative / total_true * 100
error_type2 = false_positive / total_false * 100

print(f"\nОшибка 1 рода (ушел, но не предсказали): {error_type1:.2f}%")
print(f"Ошибка 2 рода (не ушел, но предсказали): {error_type2:.2f}%")
