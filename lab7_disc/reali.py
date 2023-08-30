import math
import math

def equation(x):
    return -x**4 + 3*x**3 - 2*x + 1

def derivative(x):
    return -4 * x**3 + 9 * x**2 - 2

def bisection_method(a, b, alpha):
    while abs(b - a) > alpha:
        c = (a + b) / 2
        if abs(equation(c)) < alpha:
            return c
        elif equation(a) * equation(c) < 0:
            b = c
        else:
            a = c

    return (a + b) / 2

def chord_method(a, b, alpha):
    x_k_minus_1 = a
    x_k = b

    while abs(x_k - x_k_minus_1) > alpha:
        x_k_plus_1 = x_k - (equation(x_k) * (x_k_minus_1 - x_k)) / (equation(x_k_minus_1) - equation(x_k))
        if abs(equation(x_k_plus_1)) < alpha:
            return x_k_plus_1

        x_k_minus_1 = x_k
        x_k = x_k_plus_1

    return x_k

def newton_method(initial_guess, alpha):
    x_k_minus_1 = initial_guess
    x_k = x_k_minus_1 - equation(x_k_minus_1) / derivative(x_k_minus_1)

    while abs(x_k - x_k_minus_1) > alpha or abs(equation(x_k)) > alpha:
        x_k_minus_1 = x_k
        x_k = x_k_minus_1 - equation(x_k_minus_1) / derivative(x_k_minus_1)

    return x_k

alpha = 1e-6

root_bisection_1 = bisection_method(-1, 0, alpha)
root_chord_1 = chord_method(-1, 0, alpha)
root_newton_1 = newton_method(-1, alpha)

root_bisection_2 = bisection_method(2, 3, alpha)
root_chord_2 = chord_method(2, 3, alpha)
root_newton_2 = newton_method(3, alpha)

print("Наближені значення коренів:")
print("Метод бісекції (корінь 1):  ", root_bisection_1)
print("Метод хорд (корінь 1):      ", root_chord_1)
print("Метод Ньютона (корінь 1):   ", root_newton_1)
print("Метод бісекції (корінь 2):  ", root_bisection_2)
print("Метод хорд (корінь 2):      ", root_chord_2)
print("Метод Ньютона (корінь 2):   ", root_newton_2)
