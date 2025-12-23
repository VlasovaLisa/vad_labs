
# функция для чтения файла и подсчета писем от каждого автора
def read_authors(file_path: str) -> dict:

    authors = {} # словарь вида {email: количество писем}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f: 
            if line.startswith("From "): # строки начинающиеся с фром содержат адрес автора
                parts = line.split() # разбиваем строку на части
                if len(parts) > 1:
                    email = parts[1] # второй элемент - адрес отправителя
                    # увеличиваем счетчик писем для данного email
                    # authors.get(email, 0) возвращает текущее количество (или 0, если email ещё не встречался)
                    authors[email] = authors.get(email, 0) + 1
    return authors

# функция находит автора, написавшего больше всего писем
def find_top_author(authors: dict) -> tuple:

    max_author = None # пока автор неизвестен
    max_count = 0 # максимальное количество писем (начинаем с 0)
    for email, count in authors.items(): # проходим по всем авторам
        if count > max_count: # если нашли более активного автора
            max_count = count # обновляем максимум
            max_author = email
    return max_author, max_count # возвращаем email и его количество писем

# функция возвращает список из n самых активных авторов

    # authors.items() → список пар (email, количество)
    # key=lambda x: x[1] → сортируем по количеству писем (x[1])
    # reverse=True → сортировка по убыванию
    # [:n] → берём первые n элементов
def get_top_n(authors: dict, n: int = 10) -> list:
    
    return sorted(authors.items(), key=lambda x: x[1], reverse=True)[:n]


def main():
    file_path = "mbox.txt"
    authors = read_authors(file_path) # получаем словарь {email: количество}
 
    print("Всего авторов:", len(authors)) # выводим количество уникальных авторов

    top_author, top_count = find_top_author(authors) # ищем самого активного автора
    print("\nСамый активный автор:", top_author, "—", top_count, "писем")

    print("\nТоп-10 авторов:") # выводим список из 10 самых активных
    for email, count in get_top_n(authors, 10):
        print(email, count) # печатаем email и число писем


if __name__ == "__main__":
    main()
