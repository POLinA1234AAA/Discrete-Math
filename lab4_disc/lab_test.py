import numpy as np

def frobenius_normal_form(A):
    n = len(A)
    R = A.copy()

    for k in range(n - 1):
        for i in range(k + 1, n):
            if R[i, k] != 0:
                multiplier = R[i, k] / R[k, k]
                R[i, :] -= multiplier * R[k, :]

    return R

def eigenvalue_eigenvector(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigenvalues = np.round(eigenvalues, 5)

    return eigenvalues, eigenvectors

# Матриця А
A = np.array([[6.7, 1.14, 0.93, 1.18],
              [1.14, 3.72, 1.3, 0.16],
              [0.93, 1.3, 5.88, 2.1],
              [1.18, 0.16, 2.1, 5.66]])

# Зведення матриці А до нормальної форми Фробеніуса
R = frobenius_normal_form(A)
print("Матриця R (нормальна форма Фробеніуса):")
print(R)
print()

# Отримання власних чисел та власних векторів
eigenvalues, eigenvectors = eigenvalue_eigenvector(R)
eigenvalues = np.round(eigenvalues, 5)

print("Власні числа:")
for i, eigenvalue in enumerate(eigenvalues):
    print(f"λ{i+1} = {eigenvalue:.5f}")

print("\nВласні вектори:")
for i, eigenvector in enumerate(eigenvectors.T):
    print(f"Власний вектор для λ{i+1}:")
    print(eigenvector)

# Перевірка точності знайдених результатів
for i, eigenvalue in enumerate(eigenvalues):
    A_times_eigenvector = np.dot(A, eigenvectors[:, i])
    result = np.isclose(A_times_eigenvector, eigenvalue * eigenvectors[:, i], rtol=1e-5)
    print(f"\nПеревірка для λ{i+1}:")
    print(result)
import numpy as np

# Задана матриця A
A = np.array([[6.7, 1.14, 0.93, 1.18],
              [1.14, 3.72, 1.3, 0.16],
              [0.93, 1.3, 5.88, 2.1],
              [1.18, 0.16, 2.1, 5.66]])

# Зведення матриці A до нормальної форми Фробеніуса R
R = np.linalg.matrix_power(A, 5)  # Приклад: підносимо A до 5-ї степені
print("Нормальна форма Фробеніуса R:")
print(R)

# Отримання характеристичного рівняння та власних чисел
eigenvalues, eigenvectors = np.linalg.eig(R)
eigenvalues = np.around(eigenvalues, decimals=5)  # Заокруглюю власні числа до 5 знаків після коми
print("Власні числа λ1, λ2, ..., λm:")
print(eigenvalues)

# Знаходження власних векторів
eigenvectors_A = np.dot(np.linalg.inv(R), eigenvectors)  # Використовую  властивість A*v = λ*v
print("Власні вектори v1, v2, ..., vm:")
print(eigenvectors_A)

# Перевірка точності результатів
for i in range(len(eigenvalues)):
    Av = np.dot(A, eigenvectors_A[:, i])
    lambda_v = eigenvalues[i] * eigenvectors_A[:, i]
    result = np.allclose(Av, lambda_v, atol=1e-5)  # Перевіряю точність з точністю до 5 знаків після коми
    print(f"Перевірка для λ{i+1}: {result}")

