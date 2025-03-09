import numpy as np
import matplotlib.pyplot as plt


def sor(A, b, w, x0, tol=1e-3, max_iter=100):
    n = len(b)
    x = np.array(x0, dtype=float)
    iter_list = []
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(i))
            sum2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (1 - w) * x_old[i] + (w / A[i][i]) * (b[i] - sum1 - sum2)

        iter_list.append(np.linalg.norm(x - x_old, ord=np.inf))
        if iter_list[-1] < tol:
            return x, k + 1, iter_list
    return x, max_iter, iter_list


def gauss_seidel(A, b, x0, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.array(x0, dtype=float)
    iter_list = []
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(i))
            sum2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - sum1 - sum2) / A[i][i]

        iter_list.append(np.linalg.norm(x - x_old, ord=np.inf))
        if iter_list[-1] < tol:
            return x, k + 1, iter_list
    return x, max_iter, iter_list


A = np.array([[4, -1, 0, 0, 0, 0],
              [1, 4, -1, 0, 0, 0],
              [0, -1, 4, 0, 0, 0],
              [0, 0, 0, 4, -1, 0],
              [0, 0, 0, -1, 4, -1],
              [0, 0, 0, 0, -1, 4]], dtype=float)

b = np.array([0, 5, 0, 6, -2, 6], dtype=float)

w = 1.1  # Factor de relajación
x0 = np.zeros(len(b))  # Vector inicial

sol_sor, iter_sor, err_sor = sor(A, b, w, x0)
sol_gs, iter_gs, err_gs = gauss_seidel(A, b, x0)

print("Solución SOR:", sol_sor)
print("Iteraciones SOR:", iter_sor)
print("Solución Gauss-Seidel:", sol_gs)
print("Iteraciones Gauss-Seidel:", iter_gs)

# Graficar la convergencia
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(err_sor) + 1), err_sor, label='SOR', marker='o')
plt.plot(range(1, len(err_gs) + 1), err_gs, label='Gauss-Seidel', marker='s')
plt.yscale('log')
plt.xlabel('Iteraciones')
plt.ylabel('Error (Norma Infinito)')
plt.title('Comparación de convergencia entre SOR y Gauss-Seidel')
plt.legend()
plt.grid()
plt.show()
