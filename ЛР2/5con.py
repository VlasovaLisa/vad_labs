# дата название цена
from datetime import datetime

# класс описывает 1 заказ
class Order():

    def __init__(self, date, name, price):
        self.__date = date # дата
        self.__name = name # название пиццы
        self.__price = price # цена

    # методы для получения значений
    def get_date(self):
        return self.__date

    def get_name(self):
        return self.__name

    def get_price(self):
        return self.__price


oreder1 = Order('20-10-2005', '4 chees', 500)

# класс длс сбора всех заказов в список self.orders и выполнение анализа
class OrderAnalyze():
    def __init__(self):
        self.orders = []

    # добавление заказа в список
    def add_order(self, order):
        self.orders.append(order)

    # чтение заказов из файда
    def read_note(self, note):

        while True:

            line = note.readline()
            if not line:
                break
            order_list = line.split()
            date = datetime.strptime(order_list[0], '%Y-%m-%d')
            name = order_list[1]
            price = float(order_list[2])
            self.add_order(Order(date, name, price))

    # статистика по пиццам
    # словарь: ключ - название пиццы, значение - количество заказов
    def get_pizza_stats(self):

        pizza_stat = {}
        for order in self.orders:
            if order.get_name() in pizza_stat:
                pizza_stat[order.get_name()] += 1
            else:
                pizza_stat[order.get_name()] = 1

        return sorted(pizza_stat.items(), key=lambda x: (-x[1], x[0]))

    # печатаем таблицу
    def print_pizza_stats(self):
        pizza_stats = self.get_pizza_stats()
        print("СТАТИСТИКА ПО ПИЦЦАМ:")
        print("Пицца        | Количество заказов")
        print("---------------------------------")
        for pizza, count in pizza_stats:
            print(f"{pizza:<12} | {count}")
        print()

    # статистика по датам 
    # словарь: ключ - дата, значение - сумма цен заказов
    def get_date_stats(self):

        date_stat = {}
        for order in self.orders:
            date = order.get_date().strftime('%Y-%m-%d')
            if date in date_stat:
                date_stat[date] += order.get_price()
            else:
                date_stat[date] = order.get_price()

        return sorted(date_stat.items(), key=lambda x: x[0])

    # формируем таблицу: дата + сумма продаж в этот день
    def print_date_stats(self):
        date_stats = self.get_date_stats()
        print("СТАТИСТИКА ПО ДАТАМ:")
        print("Дата         | Общая выручка")
        print("---------------------------------")
        for date, total in date_stats:
            print(f"{date:<12} | {total:.2f} руб.")
        print()

    # самый дорогой заказ
    def get_max_price(self):

        mpo = max(self.orders, key=lambda x: x.get_price())

        return f"САМЫЙ ДОРОГОЙ ЗАКАЗ: \n|{mpo.get_date().strftime('%Y-%m-%d')} | {mpo.get_name()} | {mpo.get_price()}| \n"

    # средняя цена заказа
    def get_avg_price(self):

        avg = sum(order.get_price()
                  for order in self.orders) / len(self.orders)

        return f'СРЕДНЯЯ СТОИМОСТЬ ЗАКАЗА: \n|{avg:.2f}|'


file = open('file_vad5.txt', "r", encoding="utf-8-sig")
analyzer = OrderAnalyze()
analyzer.read_note(file)
analyzer.print_pizza_stats()
analyzer.print_date_stats()
print(analyzer.get_max_price())
print(analyzer.get_avg_price())
