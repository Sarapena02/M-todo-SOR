import numpy as np

#Se crea una función en la cual va a recibir la Matriz A y b, el factor de relajación (w) y el vector inicial (x0).
#Adicionalmente, dentro de los parametros de la función, se agrega el factor de tolerancia y el número máximo de iteraciones.

def sor(A, b, w, x0, tol=1e-6, max_iter=100):
    n = len(b) #Se comprueba la longitud de b para que coincida con la matriz A

    x = np.array(x0, dtype=float) #Se crea un arreglo para guardar las iteraciones.

    # Se crea un ciclo inicial donde se guarda las iteraciones realizadas hasta alcanzar la tolerancia.
    # Se crea un ciclo para que recorra las filas y las columnas. Y se realiza una comparación entre iteraciones.

    for k in range(max_iter):
        x_old = x.copy() #Se hace comparaciones entre iteraciones

        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(i)) #Calcula la suma de los elementos actualizados en cada iteración
            sum2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n)) #Calcula la suma de los elementos que no estan actualizados en cada iteración
            x[i] = (1 - w) * x_old[i] + (w / A[i][i]) * (b[i] - sum1 - sum2) #Actualiza el valor del arreglo. Se aplica la formula SOR. Se agrega w.

        # Calcula la normal infinito de la diferencia entre la solución actual (x) y la anterior (x_old).
        # la norma equivale al mayor valor absoluto de los cambios de las variables.
        if np.linalg.norm(x - x_old, ord=np.inf) < tol: #Si el cambio más grande entre las iteraciones sigue siendo < tol. Se considera que el metodo ha convergido
            return x, k + 1 # Si la condición de convergencia se cumple. El metodo retorna la solución x y el num de iteraciones realizadas
    return x, max_iter # Si no se alcanza la convergencia dentro de max_iter, se devuelve la mejor aproximación obtenida con el número máximo de iteraciones.

#Sistema de Ejemplo
A = np.array([[4, -1, 0, 0, 0, 0],
              [1, 4, -1, 0, 0, 0],
              [0, -1, 4, 0, 0, 0],
              [0, 0, 0, 4, -1, 0],
              [0, 0, 0, -1, 4, -1],
              [0, 0, 0, 0, -1, 4]], dtype=float)

b = np.array([0, 5, 0, 6, -2, 6], dtype=float)

w = 1.1  # Factor de relajación
x0 = np.zeros(len(b))  # Vector inicial

sol, iterations = sor(A, b, w, x0)
print("Solución:", sol)
print("Iteraciones:", iterations)
