import random

X = 2400
Y = 2400
FILENAME = "input.txt"
PROBABILITY = 0.2

def generate_life_map(x, y, filename, prob):
    print(f"Генерация поля {x}x{y} в файл {filename}...")
    try:
        with open(filename, 'w') as f:
            for _ in range(x):
                row = "".join(['#' if random.random() < prob else '.' for _ in range(y)])
                f.write(row + '\n')
        print("Готово.")
    except Exception as e:
        print(f"Ошибка при записи: {e}")

if __name__ == "__main__":
    generate_life_map(X, Y, FILENAME, PROBABILITY)