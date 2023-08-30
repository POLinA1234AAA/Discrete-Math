import numpy as np
import matplotlib.pyplot as plt

# Задані параметри
a = 1.0 + 0.4 * 4
x0 = 0
y0 = 0.5
h = 0.1
n = 61

# Функція, що задає систему диференціальних рівнянь
def f(x, y):
    return y + a * y * np.sin(x - y ** 2)

# Метод Рунге-Кутта четвертого порядку
def runge_kutta_method(f, x0, y0, h, n):
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = x0
    y[0] = y0

    for i in range(1, n):
        k1 = h * f(x[i-1], y[i-1])
        k2 = h * f(x[i-1] + h/2, y[i-1] + k1/2)
        k3 = h * f(x[i-1] + h/2, y[i-1] + k2/2)
        k4 = h * f(x[i-1] + h, y[i-1] + k3)

        x[i] = x[i-1] + h
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return x, y

# Обчислення методом Рунге-Кутта
x_runge_kutta, y_runge_kutta = runge_kutta_method(f, x0, y0, h, n)

# Графік y(x)
plt.figure(figsize=(8, 6))
plt.plot(x_runge_kutta, y_runge_kutta, label="Runge-Kutta")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Graph of y(x)")
plt.legend()
plt.grid(True)
plt.show()

# Фазовий портрет системи
plt.figure(figsize=(8, 6))
plt.plot(y_runge_kutta, y_runge_kutta, label="Phase Portrait")
plt.xlabel("y")
plt.ylabel("y'")
plt.title("Phase Portrait")
plt.legend()
plt.grid(True)
plt.show()
