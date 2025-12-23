import numpy as np
import matplotlib.pyplot as plt

# 1
# загружаем список стран из файла производства
countries = np.genfromtxt(
    'global-electricity-generation.csv', 
    delimiter = ',', 
    dtype = str, 
    skip_header = 1, 
    usecols = 0, # первый столбец (страны)
    encoding = 'utf-8' 
)

# загружаем числовые данные о производстве электроэнергии (все столбцы, кроме первого)
generation_values = np.genfromtxt(
    'global-electricity-generation.csv',
    delimiter = ',',
    skip_header = 1,
    usecols = range(1, 31), # столбцы 2–31 (по годам 1992–2021)
    encoding = 'utf-8'
)

# аналогично загружаем данные о потреблении электроэнергии
consumption_values = np.genfromtxt(
    'global-electricity-consumption.csv',
    delimiter = ',',
    skip_header = 1,
    usecols = range(1, 31),
    encoding = 'utf-8'
)

# 2 
# средние значения за последние 5 лет

# проверка, что есть хотя бы один столбец данных
if generation_values.shape[1] > 0 and consumption_values.shape[1] > 0:
    # если меньше 5 лет — берём доступные годы (все, что есть)
    num_years = min(5, generation_values.shape[1], consumption_values.shape[1])

    # берем последние `num_years` лет
    generation_last5 = generation_values[:, -num_years:]
    consumption_last5 = consumption_values[:, -num_years:]

    # считаем среднее по каждой стране, игнорируя NaN
    generation_mean = np.array([
        np.nanmean(row) if not np.isnan(row).all() else np.nan
        for row in generation_last5
    ])
    consumption_mean = np.array([
        np.nanmean(row) if not np.isnan(row).all() else np.nan
        for row in consumption_last5
    ])
else:
    # если вообще нет данных — заполняем NaN
    generation_mean = np.full(generation_values.shape[0], np.nan)
    consumption_mean = np.full(consumption_values.shape[0], np.nan)

# вывод результатов
print("\n2. Средние значения за последние 5 лет:")
for country, gen, cons in zip(countries, generation_mean, consumption_mean):
    if not np.isnan(gen) or not np.isnan(cons):
        gen_str = f"{gen:10.2f}" if not np.isnan(gen) else "   нет данных"
        cons_str = f"{cons:10.2f}" if not np.isnan(cons) else "   нет данных"
        print(f"{country:<25} производство: {gen_str} | потребление: {cons_str}")


# 3
# аналитические вычисления

# 3.1
# суммарное потребление по годам 

total_consumption_by_year = np.nansum(consumption_values, axis=0)

# формируем массив лет (начиная с 1992)
years = np.arange(1992, 1992 + total_consumption_by_year.shape[0])

print("\n3.1 Суммарное потребление по годам:")
# zip(years, total_consumption_by_year) объединяет значения лет и соответствующее потребление
for year, total in zip(years, total_consumption_by_year):
    print(f"{year}: {total:.2f}")


# 3.2 
# максимальное производство 1 страной за 1 год 

max_generation_value = np.nanmax(generation_values)

# индексы строки (страна) и столбца (год), где достигнут максимум
country_idx, year_idx = np.unravel_index(np.nanargmax(generation_values), generation_values.shape)

# массив лет (тот же, что использовался выше)
years = np.arange(1992, 1992 + generation_values.shape[1])

# определяем страну и год для максимального значения
max_country = countries[country_idx]
max_year = years[year_idx]

print("\n3.2 Максимальное производство электроэнергии:")
print(f"Значение: {max_generation_value:.2f} млрд кВт*ч")
print(f"Страна: {max_country}")
print(f"Год: {max_year}")


# 3.3 
# страны, производящие > 500 млрд кВт*ч в среднем за последние 5 лет

high_producers = countries[generation_mean > 500] # используем булевую маску
print("\n3.3 >500 млрд кВт*ч:", high_producers) # вывод массива только тех стран, где условие выполняется


# 3.4 
# 10% стран с наибольшим потреблением 

# 90 квантиль - порог, выше которого находятся 10% самых больших значений
q_90 = np.nanquantile(consumption_mean, 0.9)
top10_consumers = countries[consumption_mean >= q_90]
print("\n3.4 Топ 10% по потреблению:", top10_consumers)


# 3.5 
# страны, увеличившие производство > 10 раз с 1992 по 2021 

# берем первый столбец (1992) и последний (2021)
# проверяем, у кого значение 2021 года больше в 10 раз
in_10x = countries[generation_values[:, -1] > 10 * generation_values[:, 0]] 
print("\n3.5 Увеличили производство >10 раз:", in_10x)


# 3.6 
# страны, у которых потребление >100 и произведено меньше, чем потрачено 

# суммируем по всем годам для каждой страны
total_generation = np.nansum(generation_values, axis=1)
total_consumption = np.nansum(consumption_values, axis=1)

# создаем булевую маску и выбираем страны, удовлетворяющие обоим условиям
less_produced_more_used = countries[(total_consumption > 100) & (total_generation < total_consumption)]
print("\n3.6 Потратили >100, произвели меньше:", less_produced_more_used)


# 3.7 
# страна с максимальным потреблением в 2020 году 

year_index_2020 = -2 # индекс -2 соответствует 2020 году
max_consumption_country_2020 = countries[np.nanargmax(consumption_values[:, year_index_2020])] # возвращает индекс максимального значения
print("\n3.7 Больше всего потратила в 2020:", max_consumption_country_2020)

