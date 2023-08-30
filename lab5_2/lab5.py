import numpy as np
import matplotlib.pyplot as plt

def function(x, alpha):
    result = np.sin(alpha/2 * x)
    result[x >= 0] += np.sqrt(x[x >= 0] * alpha)
    return result

def newton_interpolation(x, y, xi):
    n = len(x)
    coefficients = y.copy()

    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coefficients[i] = (coefficients[i] - coefficients[i-1]) / (x[i] - x[i-j])

    result = coefficients[-1]
    for i in range(n-2, -1, -1):
        result = result * (xi - x[i]) + coefficients[i]

    return result

def cubic_spline_interpolation(x, y, xi):
    n = len(x)
    h = np.diff(x)
    b = np.zeros(n)
    u = np.zeros(n)
    v = np.zeros(n)
    z = np.zeros(n)
    spline_y = np.zeros_like(xi)

    for i in range(1, n-1):
        b[i] = 2 * (h[i-1] + h[i])
        u[i] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])

    for i in range(1, n-1):
        if b[i-1] != 0:
            w = h[i] / b[i-1]
            b[i] -= w * h[i-1]
            u[i] -= w * u[i-1]

    for i in range(n-2, 0, -1):
        if b[i] != 0:
            z[i] = (u[i] - h[i] * z[i+1]) / b[i]

    for i in range(1, n):
        if h[i-1] != 0:
            w = (y[i] - y[i-1]) / h[i-1] - h[i-1] * (z[i] + 2 * z[i-1]) / 6
            v[i] = w / h[i-1]

    for j in range(len(xi)):
        k = np.searchsorted(x, xi[j])
        if k == 0:
            k = 1
        elif k == n:
            k = n - 1

        h_k = x[k] - x[k-1]
        if h_k != 0:
            spline_y[j] = ((x[k] - xi[j])**3 * z[k-1] + (xi[j] - x[k-1])**3 * z[k]) / (6 * h_k) + \
                          ((x[k] - xi[j]) * y[k-1] + (xi[j] - x[k-1]) * y[k]) / h_k - \
                          h_k * ((x[k] - xi[j]) * v[k-1] + (xi[j] - x[k-1]) * v[k]) / 6

    return spline_y

# Задані точки інтерполяції
x = np.array([-2, 0, 2, 4, 6])
alpha = 2
y = function(x, alpha)
# Інтерполяційний поліном Ньютона
xi = np.linspace(-2, 6, 100)
interpolated_newton = newton_interpolation(x, y, xi)
print("Інтерполяційний поліном Ньютона:")
print(interpolated_newton)
# Інтерполяція кубічними сплайнами
xi = np.linspace(-2, 6, 100)
interpolated_spline = cubic_spline_interpolation(x, y, xi)
# Відображення результатів
plt.figure(figsize=(8, 6))
plt.plot(xi, function(xi, alpha), label='Точна функція')
plt.plot(xi, interpolated_spline, label='Інтерполяція кубічними сплайнами')
plt.scatter(x, y, color='red', label='Точки інтерполяції')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Інтерполяція кубічними сплайнами')
plt.legend()
plt.grid(True)
plt.show()

# Відображення результатів
plt.figure(figsize=(8, 6))
plt.plot(xi, function(xi, alpha), label='Точна функція')
plt.plot(xi, interpolated_newton, label='Інтерполяційний поліном Ньютона')
plt.scatter(x, y, color='red', label='Точки інтерполяції')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Інтерполяційний поліном Ньютона')
plt.legend()
plt.grid(True)
plt.show()







