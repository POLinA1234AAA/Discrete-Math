import numpy as np

def gaussian_elimination(A, b):
    n = len(A)
    m = len(A[0])
    augmented_matrix = np.column_stack((A, b))

    # Прямий хід методу Гауса
    for pivot_row in range(n):
        # Нормалізувати головний елемент до 1
        factor = augmented_matrix[pivot_row][pivot_row]
        augmented_matrix[pivot_row] /= factor

        # Відняти головний рядок від інших рядків
        for row in range(pivot_row + 1, n):
            factor = augmented_matrix[row][pivot_row]
            augmented_matrix[row] -= factor * augmented_matrix[pivot_row]

    # Зворотній хід методу Гауса
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented_matrix[i][-1]
        for j in range(i + 1, n):
            x[i] -= augmented_matrix[i][j] * x[j]

    return x

# Початкова матриця системи рівнянь
A = np.array([[3.81, 0.25, 1.28, 2.25],
              [2.25, 1.32, 6.08, 0.49],
              [5.31, 7.78, 0.98, 1.04],
              [10.89, 2.45, 3.35, 2.28]])

# Стовпець вільних членів
b = np.array([4.21,5.97 ,2.38 ,11.98 ])

# Розв'язок системи рівнянь
solution = gaussian_elimination(A, b)

# Виведення розв'язку
for i, x in enumerate(solution):
    print(f'x{i+1} = {x:.6f}')
