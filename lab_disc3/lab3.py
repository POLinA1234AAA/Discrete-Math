import numpy as np

# Задані матриця A та вектор b
A = np.array([[5.93, 1.12, 0.95, 1.32, 0.83],
              [1.12, 3.53, 2.12, 0.57, 0.91],
              [0.95, 2.12, 6.88, 1.29, 1.57],
              [1.32, 0.57, 1.29, 3.82, 1.25],
              [0.83, 0.91, 1.57, 1.25, 5.96]])

b = np.array([7.24, 3.21, 3.23, 6.25, 6])

# Зведення системи до еквівалентної форми з діагональною перевагою
n = len(b)
for k in range(n - 1):
    # Пошук максимального елемента у стовпці k
    max_index = np.argmax(np.abs(A[k:, k])) + k
    # Обмін рядками
    A[[k, max_index]] = A[[max_index, k]]
    b[[k, max_index]] = b[[max_index, k]]
    # Елімінація недіагональних елементів
    for i in range(k + 1, n):
        factor = A[i, k] / A[k, k]
        A[i, k:] -= factor * A[k, k:]
        b[i] -= factor * b[k]

# Розв'язання системи ітераційним методом
max_iterations = 100
tolerance = 1e-6

x = np.zeros(n)  # Початкове наближення розв'язку

for _ in range(max_iterations):
    x_new = np.zeros(n)
    for i in range(n):
        x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    residual = np.linalg.norm(b - np.dot(A, x_new))
    if residual < tolerance:
        x = x_new
        break
    x = x_new

# Виведення результатів
print("Матриця A з діагональною перевагою:")
print(A)

print("Вектор b:")
print(b)

print("Розв'язок системи:")
print(x)

# Обчислення нев'язки r = b - Ax
r = b - np.dot(A, x)
print("Нев'язка r:")
print(r)

import numpy as np

# Задання матриці A
A = np.array([[5.93, 1.12, 0.95, 1.32, 0.83],
              [1.12, 3.53, 2.12, 0.57, 0.91],
              [0.95, 2.12, 6.88, 1.29, 1.57],
              [1.32, 0.57, 1.29, 3.82, 1.25],
              [0.83, 0.91, 1.57, 1.25, 5.96]])

# Задання вектора b
b = np.array([7.24, 3.21, 3.23, 6.25, 6.00])

# Зведення матриці A до еквівалентної форми з діагональною перевагою
D = np.diag(np.diag(A))
L = np.tril(A, k=-1)
U = np.triu(A, k=1)
A_eq = D - L - U

# Виведення матриці A з діагональною перевагою
print("Матриця A з діагональною перевагою:")
np.set_printoptions(precision=2, suppress=True)
print(A_eq)

# Розв'язок системи за ітераційним методом
n = len(b)
x = np.zeros(n)
tolerance = 1e-6  # Точність розв'язку
max_iterations = 100  # Максимальна кількість ітерацій

for iteration in range(max_iterations):
    x_new = np.zeros(n)
    for i in range(n):
        x_new[i] = (b[i] - np.dot(A_eq[i, :i], x_new[:i]) - np.dot(A_eq[i, i + 1:], x[i + 1:])) / A_eq[i, i]
    if np.linalg.norm(x - x_new, ord=np.inf) < tolerance:
        break
    x = x_new

# Виведення розв'язку системи з двома цифрами після крапки
print("Розв'язок системи:")
np.set_printoptions(precision=2, suppress=True)
print(x)

# Обчислення нев'язки r = b - Ax
r = b - np.dot(A, x)
print("Нев'язка r:")
np.set_printoptions(precision=2, suppress=True)
print(r)
