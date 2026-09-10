# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 16:03:58 2025

@author: Natalia
"""

import matplotlib.pyplot as plt
import numpy as np

# Exemplo de definição (pode ter quantos trapézios quiser)
conjunto = {
    "trapezio": {
        "Low": (0, 0, 2, 4),
        "Medium": (2, 4, 6, 8),
        "High": (6, 8, 10, 12),
        "Very High": (10, 12, 14, 16)
    }
}

# Função trapezoidal
def trapezio(x, a, b, c, d):
    return np.maximum(
        np.minimum(
            np.minimum((x - a) / (b - a + 1e-9), 1),  # evita divisão por zero
            (d - x) / (d - c + 1e-9)
        ),
        0
    )

# Define o intervalo de x dinamicamente (pega o mínimo e máximo dos vértices)
todos_vertices = [v for t in conjunto["trapezio"].values() for v in t]
x_min, x_max = min(todos_vertices), max(todos_vertices)
x = np.linspace(x_min, x_max, 500)

# Percorre cada trapézio e plota
for nome, (a, b, c, d) in conjunto["trapezio"].items():
    y = trapezio(x, a, b, c, d)
    plt.plot(x, y, label=nome)         # linha com label
    plt.fill_between(x, y, alpha=0.2)  # área preenchida

# Configuração do gráfico
plt.ylim(-0.1, 1.1)
plt.xlabel("x")
plt.ylabel("µ(x)")
plt.title("Funções Fuzzy - Trapézios")
plt.grid(True)
plt.legend()
plt.show()
