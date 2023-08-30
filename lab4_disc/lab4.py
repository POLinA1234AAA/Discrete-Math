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
R = np.round(R, 5)
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
    A_times_eigenvector = np.round(A_times_eigenvector, 5)
    result = np.isclose(A_times_eigenvector, eigenvalue * eigenvectors[:, i], rtol=1e-5)
    print(f"\nПеревірка для λ{i+1}:")
    print(result)

