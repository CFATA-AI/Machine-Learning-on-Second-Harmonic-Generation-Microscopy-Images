# Preprocessing

import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, gaussian_kde, mode

start_time = time.time()

carpeta = r'C:\DATA'

if os.path.exists(carpeta):
    os.chdir(carpeta)
else:
    print(f"La ruta {carpeta} no existe.")

archivos = glob.glob("*.txt")
num = len(archivos)

F1, F2, F3, F4 = [], [], [], []

for l in range(num):
    nombre_archivo = archivos[l]
    data = np.loadtxt(nombre_archivo)

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2].copy()

    a = x[-1] - x[0]
    b = x[1] - x[0]
    e = int(round(a / b + 1))

    a1 = y[-1] - y[0]
    b1 = y[e] - y[0]
    e1 = int(round(a1 / b1 + 1))

    # Reshape original image
    r1 = z.reshape((e1, e)).T

    # Remove extreme maxima
    ar = 300
    ordena = np.sort(z)[::-1]
    maximos = ordena[:ar]
    ubicaciones = np.isin(z, maximos)

    for i in range(ar - 1):
        if maximos[i] - maximos[i + 1] > 5:
            zmax = np.max(z)
            logical_values = (z == zmax)
            z[logical_values] = 0
            z[logical_values] = np.max(z)

    r2 = z.reshape((e1, e)).T
    ww = ubicaciones.reshape((e1, e)).T.astype(int)

    # Noise threshold
    noise_threshold = np.median(z) + 3 * np.std(z)
    threshold_15 = 0.15 * noise_threshold

    t = r2.shape

    MEDIA, VAR, ASI, CURT = [], [], [], []
    MODE, TOTAL_INT, PERC_HIGH = [], [], []

    r = 20
    s = 19
    lim = 0
    medidas1 = 0

    # Sliding window
    for i in range(t[0] - r):
        for j in range(t[1] - r):

            mat = r2[i:i + s + 1, j:j + s + 1]
            nueva = ww[i:i + s + 1, j:j + s + 1]

            if np.sum(nueva) <= lim:
                B = mat.flatten()

                MEDIA.append(np.mean(B))
                VAR.append(np.var(B, ddof=0))
                ASI.append(skew(B))
                CURT.append(kurtosis(B, fisher=False))

                # New features
                MODE.append(mode(B, keepdims=False).mode)
                TOTAL_INT.append(np.sum(B))
                PERC_HIGH.append(100.0 * np.sum(B > threshold_15) / len(B))

                medidas1 += 1

    if medidas1 == 0:
        prommed = promvar = promasi = promkurt = 0
        prommode = promtotal = promperc = 0
        f1 = f2 = f3 = f4 = 0
    else:
        prommed = np.mean(MEDIA)
        promvar = np.mean(VAR)
        promasi = np.mean(ASI)
        promkurt = np.mean(CURT)

        prommode = np.mean(MODE)
        promtotal = np.mean(TOTAL_INT)
        promperc = np.mean(PERC_HIGH)

        # KDE
        def get_kde(data):
            if len(set(data)) <= 1:
                return np.zeros(len(data))
            kde = gaussian_kde(data)
            return kde(data)

        f1 = get_kde(MEDIA)
        f2 = get_kde(VAR)
        f3 = get_kde(ASI)
        f4 = get_kde(CURT)

    F1.append(f1)
    F2.append(f2)
    F3.append(f3)
    F4.append(f4)

    # Individual graphs
    nombre_base = os.path.splitext(nombre_archivo)[0]

    plt.figure()
    plt.imshow(ww, cmap='hot', aspect='auto')
    plt.title(f'Área evitada {nombre_base}')
    plt.savefig(f'{nombre_base}_area.jpg')
    plt.close()

    plt.figure()
    plt.imshow(r1, aspect='auto')
    plt.title(f'Imagen {nombre_base}')
    plt.savefig(f'{nombre_base}_imagen.jpg')
    plt.close()

    # Save averages
    with open('Promedios.txt', 'a') as f:
        f.write(
            f"{prommed:.17e} {promvar:.17e} {promasi:.17e} {promkurt:.17e} "
            f"{prommode:.17e} {promtotal:.17e} {promperc:.17e}\n"
        )

# Final graphs
titulos = ['Primer momento', 'Segundo momento', 'Tercer momento', 'Cuarto momento']
funciones = [F1, F2, F3, F4]

for idx, func_list in enumerate(funciones):
    plt.figure()
    for f in func_list:
        if isinstance(f, np.ndarray):
            plt.plot(f)
    plt.title(titulos[idx])
    plt.savefig(f'Grafica_{titulos[idx]}.jpg')
    plt.close()

print(f"Tiempo total: {time.time() - start_time:.4f} segundos")
