import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# функция для вывода метрик для 15
def prt_m(y_true, y_pred, title=""):
    print(f"\n{title}:")
    print("Precision:", round(precision_score(y_true, y_pred), 3))
    print("Recall:", round(recall_score(y_true, y_pred), 3))
    print("F1-score:", round(f1_score(y_true, y_pred), 3))




# 1
# загрузка и чтение файла
data = pd.read_csv('База.csv', delimiter=';', encoding='cp1251')
pd.set_option('display.max_columns', None)  # показывать все столбцы
# вывод первых 5 строк датасета
print(data.head())



# 2
# предварительная фильтрация
# фильтруем только жилые помещения
data = data[data['ВидПомещения'] == 'жилые помещения']

# фильтруем строки с определенным статусом
# оставляем только те, где СледующийСтатус != пусто и != "В резерве"
data = data[~data['СледующийСтатус'].isin(['', 'В резерве'])]

# преобразуем целевой признак в числовой: Продана -> 1, Свободна -> 0
data['target'] = data['СледующийСтатус'].map({'Продана': 1, 'Свободна': 0})

# удаляем ненужные столбцы
data = data.drop(columns=['УИД_Брони', 'ВидПомещения','ВремяБрони', 'ДатаБрони', 'СледующийСтатус'])

# проверяем результат
print("\n2 Предварительная фильтрация-----------------------------")
print(data.head())
print("\nКоличество строк после фильтрации:", len(data))



# 3
# преобразование типов данных

# Числовые колонки
num_cols = ['ПродаваемаяПлощадь', 'ФактическаяСтоимостьПомещения', 'СкидкаНаКвартиру']
for col in num_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Бинарные признаки
bin_cols = {
    'ИсточникБрони': {'МП': 1, 'ручная': 0},
    'ВременнаяБронь': {'Да': 1, 'Нет': 0},
    'ТипСтоимости': {'Стоимость при 100% оплате': 1, 'Стоимость в рассрочку': 0},
    'ВариантОплаты': {'Единовременная оплата': 1, 'Оплата в рассрочку': 0},
    'СделкаАН': {'Да': 1, 'Нет': 0},
    'ИнвестиционныйПродукт': {'Да': 1, 'Нет': 0},
    'Привилегия': {'Да': 1, 'Нет': 0}
}
for col, mapping in bin_cols.items():
    data[col] = data[col].map(mapping)

# Категориальные -> One-Hot
data['Статус лида (из CRM)'] = data['Статус лида (из CRM)'].fillna('N')
data = pd.get_dummies(data, columns=['Город', 'Статус лида (из CRM)'], drop_first=True)

# Обработка поля "Тип" (1к -> 1.0)
def convert_rooms(value):
    if isinstance(value, str) and value.endswith('к'):
        return pd.to_numeric(value[:-1].replace(',', '.'), errors='coerce')
    return np.nan
data['Тип'] = data['Тип'].apply(convert_rooms)

# Преобразуем в строку, заменяем ',' на '.' и приводим к float
data['СтоимостьНаДатуБрони'] = data['СтоимостьНаДатуБрони'].astype(str).str.replace(',', '.', regex=False)
data['СтоимостьНаДатуБрони'] = pd.to_numeric(data['СтоимостьНаДатуБрони'], errors='coerce')


# Находим все булевы столбцы
bool_cols = data.select_dtypes(include='bool').columns
# Меняем тип с bool → int
data[bool_cols] = data[bool_cols].astype('int8')

# проверка типов данных 
print("\n3 Преобразование типов данных-----------------------------")
print(data.dtypes)
# проверка первых строк
print(data.head())


# 4
# отсутствующие данные
print("\n4 Отсутствующие данные-----------------------------")
print("Количество пропусков по каждому признаку:") # проверка пропусков 
print(data.isna().sum())

# замена пропусков на 0 в СкидкаНаКвартиру
data['СкидкаНаКвартиру'] = data['СкидкаНаКвартиру'].fillna(0)

# Тип и ПродаваемаяПлощадь пропуски заменяем на медиану
data['Тип'] = data['Тип'].fillna(data['Тип'].median())
data['ПродаваемаяПлощадь'] = data['ПродаваемаяПлощадь'].fillna(data['ПродаваемаяПлощадь'].median())

# удаляем столбец ВариантОплатыДоп (по условию)
data = data.drop(columns=['ВариантОплатыДоп'])

# удаляем строки, гле не известно квартира продана или свободна
# лучший вариант для обучения модели 
data = data.dropna(subset=['target'])


# Преобразуем колонку в float
data['ФактическаяСтоимостьПомещения'] = pd.to_numeric(
    data['ФактическаяСтоимостьПомещения'], errors='coerce'
)
# Заполняем пропуски медианой
data['ФактическаяСтоимостьПомещения'] = data['ФактическаяСтоимостьПомещения'].fillna(
    data['ФактическаяСтоимостьПомещения'].median()
)

# остальные признаки
missing = data.isna().sum() # смотрим где мало пропусков
few_missing_cols = missing[(missing > 0) & (missing < 10)].index
print("Признаки с небольшим количеством пропусков:", few_missing_cols)

data = data.dropna(subset=few_missing_cols) # удаляем строки где пропусков не много

# проверка пропусков
print("Пропуски после обработки:")
print(data.isna().sum())



# 5
# дополнение данных
print("\n5 Дополнение данных-----------------------------")

data['ЦенаЗаКвадратныйМетр'] = data['ФактическаяСтоимостьПомещения'] / data['ПродаваемаяПлощадь']
data['СкидкаВПроцентах'] = (data['СкидкаНаКвартиру'] / data['ФактическаяСтоимостьПомещения']).fillna(0) * 100

# проверка первых строк
print(data.head())



# 6
# нормализация 
print("\n6 Нормализация-----------------------------")


original_discount = data['СкидкаНаКвартиру'].copy()

scaler = MinMaxScaler()
num_cols_to_scale = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
num_cols_to_scale = [col for col in num_cols_to_scale if col not in ['СледующийСтатус', 'СкидкаНаКвартиру']]
data[num_cols_to_scale] = scaler.fit_transform(data[num_cols_to_scale])

discount_scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
data['СкидкаНаКвартиру'] = discount_scaler.fit_transform(original_discount.values.reshape(-1, 1))

# Проверка первых строк
print(data.head())



# 7
# сбалансированность
print("\n7 Сбалансированность-----------------------------")

# Проверка распределения целевого признака
target_counts = data['target'].value_counts()
print(target_counts)

# Проверка балансировки
if abs(target_counts[0] - target_counts[1]) / target_counts.sum() < 0.1:
    print("Датасет приблизительно сбалансирован")
else:
    print("Датасет несбалансирован")
# один из классов доминирует по количеству над другим (продано/свободно)
    

# 8
# факторы и целевой
print("\n8 Факторы и целевая переменная-----------------------------")
factor_feat = data.columns.difference(['target'])  # все колонки, кроме целевой
target_feat = 'target'  # целевая переменная

print("\nФакторные признаки:")
print(factor_feat.tolist())
print("\nЦелевой признак:")
print(target_feat)

# 9
# разбиение датасета
print("\n9 Разбиение датасета-----------------------------")
# матрица признаков (X) и целевая переменная (y)
X = data[factor_feat]
y = data[target_feat]

# разбиение: 80% обучение, 20% тест
# random_state фиксирует случайность для воспроизводимости
# stratify=y сохраняет пропорцию классов target в обеих выборках
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# проверка размеров выборок
print("\nРазмер обучающей выборки X_train:", X_train.shape)
print("Размер тестовой выборки X_test:", X_test.shape)
print("Распределение классов в обучающей выборке:")
print(y_train.value_counts())
print("Распределение классов в тестовой выборке:")
print(y_test.value_counts())


# 10
# KNN модель
print("\n10 KNN модель-----------------------------")

# инициализация модели KNN с параметрами по умолчанию
knn_model = KNeighborsClassifier()
# обучение модели на обучающих данных
knn_model.fit(X_train, y_train)
# предсказания для обучающей выборки
knn_train_pred = knn_model.predict(X_train)
# предсказания для тестовой выборки
knn_test_pred = knn_model.predict(X_test)

print("Предсказания для обучающей выборки (первые 20 строк):")
print(knn_train_pred[:20])
print("\nПредсказания для тестовой выборки (первые 20 строк):")
print(knn_test_pred[:20])


# 11
# Decision Tree
# труктура в виде дерева, которая делит данные на группы, чтобы как можно точнее предсказать целевую переменную
print("\n11 Decision Tree модель-----------------------------")

# инициализация модели дерева решений с параметрами по умолчанию
dt_model = DecisionTreeClassifier()
# обучение модели на обучающих данных
dt_model.fit(X_train, y_train)
# предсказания для обучающей выборки
dt_train_pred = dt_model.predict(X_train)
# предсказания для тестовой выборки
dt_test_pred = dt_model.predict(X_test)

# вывод первых 20 предсказаний для проверки
print("Предсказания Decision Tree для обучающей выборки (первые 20):")
print(dt_train_pred[:20])
print("\nПредсказания Decision Tree для тестовой выборки (первые 20):")
print(dt_test_pred[:20])


# 12
# векторы прогнозных значений для каждой модели
print("\n12 Векторы прогнозов-----------------------------")

# KNN
knn_train_pred_vector = knn_train_pred  # обучающая выборка
knn_test_pred_vector = knn_test_pred    # тестовая выборка

# Decision Tree
dt_train_pred_vector = dt_train_pred    # обучающая выборка
dt_test_pred_vector = dt_test_pred      # тестовая выборка

# вывод первых 20 значений для проверки
print("KNN - обучающая выборка (первые 20):", knn_train_pred_vector[:20])
print("KNN - тестовая выборка (первые 20):", knn_test_pred_vector[:20])
print("Decision Tree - обучающая выборка (первые 20):", dt_train_pred_vector[:20])
print("Decision Tree - тестовая выборка (первые 20):", dt_test_pred_vector[:20])


# 13 
# показатели качества для моделей
print("\n13 Показатели качества-----------------------------")

models = {
    "KNN": (knn_train_pred_vector, knn_test_pred_vector),
    "Decision Tree": (dt_train_pred_vector, dt_test_pred_vector)
}

for model_name, (train_pred, test_pred) in models.items():
    print(f"\nМодель: {model_name}")
    
    # обучающая выборка
    precision_train = precision_score(y_train, train_pred)
    recall_train = recall_score(y_train, train_pred)
    f1_train = f1_score(y_train, train_pred)
    
    print("Обучающая выборка:")
    print(f"Precision: {precision_train:.3f}, Recall: {recall_train:.3f}, F1-score: {f1_train:.3f}")
    
    # тестовая выборка
    precision_test = precision_score(y_test, test_pred)
    recall_test = recall_score(y_test, test_pred)
    f1_test = f1_score(y_test, test_pred)
    
    print("Тестовая выборка:")
    print(f"Precision: {precision_test:.3f}, Recall: {recall_test:.3f}, F1-score: {f1_test:.3f}")



# 14
# выводы:
    
# Precision (точность) показывает, какая часть объектов, предсказанных как «Продана», действительно была продана.
# KNN: 0.786 → примерно 79% предсказанных «Продана» верны.
# Decision Tree: 0.819 → примерно 82% предсказанных «Продана» оказались верными.

# Recall (полнота) показывает, какая часть всех реально проданных квартир была правильно предсказана.
# KNN: 0.711 → модель выявила только 71% реально проданных квартир.
# Decision Tree: 0.794 → модель нашла почти 79% всех реально проданных квартир.


# KNN
# Модель KNN показывает умеренно хорошие результаты.
# Precision выше, чем Recall, что означает, что модель реже ошибается 
#       в прогнозе "Продана", но иногда пропускает положительные случаи.
# F1-score на тестовой выборке ниже, чем на обучающей, что свидетельствует о небольшой потере качества.
    

# Decision Tree
# Дерево решений отлично "запомнило" данные обучающей выборки (почти идеальная точность и полнота).
# На тестовой выборке метрики падают, но остаются выше, чем у KNN по F1-score, 
#       что говорит о лучшей способности выявлять положительные случаи.
# Наблюдается легкое переобучение, так как разница между обучением и тестом значительна.
    

# Decision Tree лучше балансирует Precision и Recall, выявляя больше реально проданных квартир, 
#       при этом не делая слишком много ложных срабатываний.
    
# KNN немного осторожнее, пропуская часть положительных случаев, но делая меньше ошибок на отрицательных.


# 15
# Boxplot и удаление выбросов 
print("\n15 Boxplot-----------------------------")

num_cols_plot = ['ПродаваемаяПлощадь','СтоимостьНаДатуБрони','СкидкаНаКвартиру',
                 'ФактическаяСтоимостьПомещения','ЦенаЗаКвадратныйМетр','СкидкаВПроцентах','Тип']

plt.figure(figsize=(12,6))
sns.boxplot(data=data[num_cols_plot])
plt.xticks(rotation=45)
plt.title("Boxplot числовых признаков")
plt.show()

# удаление выбросов по IQR
Q1 = data[num_cols_plot].quantile(0.25)
Q3 = data[num_cols_plot].quantile(0.75)
IQR = Q3 - Q1
data_no_outliers = data[~((data[num_cols_plot] < (Q1 - 1.5*IQR)) | (data[num_cols_plot] > (Q3 + 1.5*IQR))).any(axis=1)]

print("\nРазмер данных после удаления выбросов:", data_no_outliers.shape)

# повторное обучение
X_filt = data_no_outliers[factor_feat]
y_filt = data_no_outliers[target_feat]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_filt, y_filt, test_size=0.2, random_state=42, stratify=y_filt
)

# переобучение моделей
knn_model.fit(X_train_f, y_train_f)
dt_model.fit(X_train_f, y_train_f)

# предсказания
knn_train_pred_f = knn_model.predict(X_train_f)
knn_test_pred_f = knn_model.predict(X_test_f)

dt_train_pred_f = dt_model.predict(X_train_f)
dt_test_pred_f = dt_model.predict(X_test_f)

# метрики после фильтрации 
print("\nМетрики после фильтрации выбросов:")

prt_m(y_train_f, knn_train_pred_f, "KNN обучающая выборка")
prt_m(y_test_f, knn_test_pred_f, "KNN тестовая выборка")

prt_m(y_train_f, dt_train_pred_f, "Decision Tree обучающая выборка")
prt_m(y_test_f, dt_test_pred_f, "Decision Tree тестовая выборка")


# 16
# Подбор параметров
print("\n16 Подбор параметров-----------------------------")

# KNN: подбор k 
k_range = range(1, 41)
f1_knn = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    f1_knn.append(f1_score(y_test, knn.predict(X_test)))

plt.figure(figsize=(10,4))
plt.plot(k_range, f1_knn, marker='o')
plt.title("F1-score vs k (KNN)")
plt.xlabel("Количество соседей k")
plt.ylabel("F1-score")
plt.grid(True)
plt.show()

# оптимальное k по максимальному F1
optimal_k = k_range[f1_knn.index(max(f1_knn))]
print("Оптимальное k для KNN:", optimal_k)
print("Максимальный F1-score на тестовой выборке:", round(max(f1_knn), 3))


# Decision Tree: подбор глубины
depth_range = range(2, 41)
f1_dt = []

for d in depth_range:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    f1_dt.append(f1_score(y_test, dt.predict(X_test)))

plt.figure(figsize=(10,4))
plt.plot(depth_range, f1_dt, marker='o', color='red')
plt.title("F1-score vs глубина дерева (Decision Tree)")
plt.xlabel("Глубина дерева")
plt.ylabel("F1-score")
plt.grid(True)
plt.show()

# оптимальная глубина дерева
optimal_depth = depth_range[f1_dt.index(max(f1_dt))]
print("Оптимальная глубина дерева для Decision Tree:", optimal_depth)
print("Максимальный F1-score на тестовой выборке:", round(max(f1_dt), 3))

# KNN
# Оптимальное k = 7. Это означает, что для классификации каждой квартиры алгоритм учитывает 7 ближайших соседей.
#       Чем больше k, тем сглаженнее решение (меньше переобучения, но может упрощать модель).
#       Меньшее k — модель более чувствительна к шуму.
# Максимальный F1-score = 0.773. На тестовой выборке это средневзвешенная метрика точности и полноты,
#       показывающая баланс между правильными положительными предсказаниями и полнотой обнаружения проданных квартир.

# Decision Tree
# Оптимальная глубина = 15. Глубина дерева ограничивает, сколько уровней решений может принимать модель.
#       15 позволяет дереву учитывать достаточное число признаков для точного разделения классов, но не слишком переобучаться.
# Максимальный F1-score = 0.82.Значение выше, чем у KNN, значит, что дерево решений на тестовой выборке 
#       точнее классифицирует квартиры как проданные или свободные.

# Decision Tree работает лучше, чем KNN для этого набора данных, 
#       потому что её максимальный F1-score выше (0.82 против 0.773).


# 17
# Логическая регрессия
print("\n17 Логическая регрессия-----------------------------")

# инициализация модели
log_model = LogisticRegression(max_iter=1000)
# обучение модели
log_model.fit(X_train, y_train)

# предсказания
log_train_pred = log_model.predict(X_train)
log_test_pred = log_model.predict(X_test)

# метрики
prt_m(y_train, log_train_pred, "Logistic Regression train")
prt_m(y_test, log_test_pred, "Logistic Regression test")

# вывод сравнения
print("\nСравнение моделей на тестовой выборке:")
print(f"KNN test F1-score: {f1_score(y_test, knn_test_pred):.3f}")
print(f"Decision Tree test F1-score: {f1_score(y_test, dt_test_pred):.3f}")
print(f"Logistic Regression test F1-score: {f1_score(y_test, log_test_pred):.3f}")

# KNN
# Модель достаточно проста, показывает средние показатели. 
# На тестовой выборке немного теряет в качестве, чувствительна к шуму и выбросам.

# Decision Tree
# Дерево решений сильно переобучилось: идеальные показатели на обучении, но тестовая выборка показывает заметное падение.
# Модель очень гибкая и легко подстраивается под обучающие данные, но хуже обобщает.

# Logistic Regression
# Логистическая регрессия показывает наиболее сбалансированные метрики между Precision и Recall.
# Модель меньше подвержена переобучению, хорошо обобщает данные. На тестовой выборке F1 выше, чем у KNN и Decision Tree.

# Для прогнозирования продажи квартир на этом датасете Logistic Regression оказалась лучшей моделью,
# так как она дает высокий F1-score на тестовой выборке и сбалансированные показатели Precision и Recall.



# 18
# Linear SVM 
# алгоритм классификации, который пытается найти границу (линию, плоскость или гиперплоскость),
# которая лучше всего разделяет на 2 класса
print("\n18 inear SVM -----------------------------")

# инициализация модели LinearSVC с увеличенным числом итераций для сходимости
svm_model = LinearSVC(max_iter=10000)

# обучение модели на обучающих данных
svm_model.fit(X_train, y_train)

# предсказания
svm_train_pred = svm_model.predict(X_train)
svm_test_pred = svm_model.predict(X_test)

# метрики качества
prt_m(y_train, svm_train_pred, "LinearSVC train")
prt_m(y_test, svm_test_pred, "LinearSVC test")

# сравнение с предыдущими моделями
print("\nСравнение моделей на тестовой выборке (F1-score):")
print(f"KNN test F1-score: {f1_score(y_test, knn_test_pred):.3f}")
print(f"Decision Tree test F1-score: {f1_score(y_test, dt_test_pred):.3f}")
print(f"Logistic Regression test F1-score: {f1_score(y_test, log_test_pred):.3f}")
print(f"Linear SVM test F1-score: {f1_score(y_test, svm_test_pred):.3f}")


# Наилучшее качество на тестовой выборке показала Linear SVM (F1 = 0.809).
# Логистическая регрессия почти равна по качеству (F1 = 0.807) и более интерпретируема.
# Decision Tree хорошо подходит для обучения, но имеет высокий риск переобучения.
# KNN показал наименее точные прогнозы.


# Linear SVM и Logistic Regression — самые стабильные и надежные модели для этой задачи прогнозирования
