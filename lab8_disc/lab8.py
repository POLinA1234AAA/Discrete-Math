import numpy as np
import matplotlib.pyplot as plt

# Функція, що задає диференціальне рівняння
def f(x, y):
    return y + a * y * np.sin(x - y**2)

# Функція, що обчислює метод Рунге-Кутта четвертого порядку
def runge_kutta_method(f, x0, y0, h, n):
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x[0] = x0
    y[0] = y0

    for i in range(n):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h/2, y[i] + k1/2)
        k3 = h * f(x[i] + h/2, y[i] + k2/2)
        k4 = h * f(x[i] + h, y[i] + k3)

        x[i+1] = x[i] + h
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return x, y

# Функція, що обчислює метод Адамса
def adams_method(f, x0, y0, h, n):
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x_runge_kutta, y_runge_kutta = runge_kutta_method(f, x0, y0, h, 3)

    # Визначення початкових точок методом Рунге-Кутта
    for i in range(4):
        x[i] = x_runge_kutta[i]
        y[i] = y_runge_kutta[i]

    # Використання методу Адамса
    for i in range(3, n):
        y[i+1] = y[i] + h * (55 * f(x[i], y[i]) - 59 * f(x[i-1], y[i-1]) + 37 * f(x[i-2], y[i-2]) - 9 * f(x[i-3], y[i-3])) / 24
        x[i+1] = x[i] + h

    return x, y

# Функція, що обчислює точний розв'язок
def true_solution(x):
    return np.exp(-np.sin(x**2))

# Задані параметри
a = 1.0 + 0.4 * 4
x0 = 0
y0 = 0.5
h = 0.1
n = 60

# Визначення точок на інтервалі
x_range = np.linspace(x0, x0 + h * n, n+1)

# Обчислення методом Рунге-Кутта
x_runge_kutta, y_runge_kutta = runge_kutta_method(f, x0, y0, h, n)

# Обчислення методом Адамса
x_adams, y_adams = adams_method(f, x0, y0, h, n)

# Обчислення точного розв'язку
y_true = true_solution(x_range)

# Обчислення функції помилки
error_runge_kutta = np.abs(y_runge_kutta - y_true)
error_adams = np.abs(y_adams - y_true)

# Побудова графіків
plt.figure(figsize=(12, 6))

# Графік наближеного розв'язку
plt.subplot(1, 2, 1)
plt.plot(x_runge_kutta, y_runge_kutta, label='Runge-Kutta')
plt.plot(x_adams, y_adams, label='Adams')
plt.plot(x_range, y_true, label='True Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Approximate Solutions')
plt.legend()

# Графік функції помилки
plt.subplot(1, 2, 2)
plt.plot(x_runge_kutta, error_runge_kutta, label='Runge-Kutta')
plt.plot(x_adams, error_adams, label='Adams')
plt.xlabel('x')
plt.ylabel('Error')
plt.title('Error')
plt.legend()

plt.tight_layout()
plt.show()

# Виведення значень наближеного розв'язку
print("\nМетод Рунге-Кутта:")
for i in range(n):
    print(f"x = {x_runge_kutta[i]:.4f}, y = {y_runge_kutta[i]:.4f}")

print("\nМетод Адамса:")
for i in range(n):
    print(f"x = {x_adams[i]:.4f}, y = {y_adams[i]:.4f}")
