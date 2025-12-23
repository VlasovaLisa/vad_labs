# считаем участников
participants = input().split() # читаем 1 строку имена через пробел и разбиваем в список
n = int(input()) # число покупок

# словарь расходов
expenses = {name: 0 for name in participants} # для каждого участника заводим начальную сумму 0

for _ in range(n):
    name, amount = input().split() # читаем строку с именем и суммой
    amount = int(amount) # преобразуем сумму в int 
    expenses[name] += amount # добавляем в соответствующее значение имени в словаре сумму покупок

total = sum(expenses.values()) # общая сумма всех расходов
k = len(participants) # количество участников
avg = total / k # средняя трата типа float

# считаем баланс, создаем словарь где имя <-> (сколько потратил - среднее значение расходов)
# если баланс положит -> человек потратил больше денег -> ему должны вернуть деньги
# если баланс отриц -> он потратил меньше среднего -> он должен доплатить другим 
# если баланс = 0 -> он потратил среднее -> николму не должен
balance = {name: expenses[name] - avg for name in participants}

# списки должников и кредиторов
debtors = [(name, -bal) for name, bal in balance.items() if bal < -1e-9]
creditors = [(name, bal) for name, bal in balance.items() if bal > 1e-9]

# сортируем по убыванию суммы долга/кредита
# жадный подход - берем сначала самых больших должников и самых больших кредиторов
debtors.sort(key=lambda x: x[1], reverse=True)
creditors.sort(key=lambda x: x[1], reverse=True)

# список переводов
transfers = []
i, j = 0, 0 # указатели на текущего должника и кредитора

while i < len(debtors) and j < len(creditors):
    debtor, debt = debtors[i]
    creditor, credit = creditors[j]
    
    # максимальный перевод между этими двумя, 
    # который либо полностью покроет долг должника, 
    # либо полностью закроет требование кредитора
    amount = min(debt, credit)
    
    transfers.append((debtor, creditor, amount))
    
    # после перевода уменьшаем значения
    debt -= amount
    credit -= amount
    
    # если кто то стал нулем - переходим к следующему (увеличиваем показатель)
    # иначе обновляем текущую запись с уменьшенной суммой

    # обновляем списки
    if debt <= 1e-9:
        i += 1
    else:
        debtors[i] = (debtor, debt)
    
    if credit <= 1e-9:
        j += 1
    else:
        creditors[j] = (creditor, credit)

    # каждая итерация добавляет 1 перевод в список transfers

# вывод
print(len(transfers))
for debtor, creditor, amount in transfers:
    print(f"{debtor} {creditor} {amount:.2f}") # кто платит, кому и сумма (.2f - 2знака после запятой, копейки)