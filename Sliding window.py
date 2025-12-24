#Preprocessing

import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, gaussian_kde

start_time = time.time()

carpeta = r'C:\data'

if os.path.exists(carpeta):
    os.chdir(carpeta)
else:
    print(f"La ruta {carpeta} no existe.")

archivos = glob.glob("*.txt")
num = len(archivos)


F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11 = [], [], [], [], [], [], [], [], [], [], []
MEAN_F1, MEAN_F2, MEAN_F3, MEAN_F4 = [], [], [], []

for l in range(num):
    nombre_archivo = archivos[l]
    data = np.loadtxt(nombre_archivo)
    x, y, z = data[:, 0], data[:, 1], data[:, 2].copy()
    
    a, b = x[-1] - x[0], x[1] - x[0]
    e = int(round(a / b + 1))
    a1, b1 = y[-1] - y[0], y[e] - y[0] 
    e1 = int(round(a1 / b1 + 1))
    
    r1 = z.reshape((e1, e)).T
    
    ar = 300
    ordena = np.sort(z)[::-1] 
    maximos = ordena[:ar]
    ubicaciones = np.isin(z, maximos)
    
    for i in range(ar - 1):
        if maximos[i] - maximos[i+1] > 5:
            zmax = np.max(z)
            z[z == zmax] = 0
            z[z == zmax] = np.max(z)
            
    r2 = z.reshape((e1, e)).T
    ww = ubicaciones.reshape((e1, e)).T.astype(int)
    
    t = r2.shape
    
    MEDIA, VAR, ASI, CURT, SUMA, PORCENTAJE = [], [], [], [], [], []
    r, s, lim, medidas1 = 20, 19, 0, 0

    # Sliding window 
    for i in range(t[0] - r):
        for j in range(t[1] - r):
            mat = r2[i : i+s+1, j : j+s+1]
            nueva = ww[i : i+s+1, j : j+s+1]
            
            if np.sum(nueva) <= lim:
                B = mat.flatten()
                m_v = np.mean(B)
                MEDIA.append(m_v)
                VAR.append(np.var(B, ddof=0)) 
                ASI.append(skew(B))
                CURT.append(kurtosis(B, fisher=False))
                SUMA.append(np.sum(B))
                
                # Porcentaje > 85%
                u = m_v * 0.85
                PORCENTAJE.append((np.sum(B > u) / len(B)) * 100)
                medidas1 += 1

    if medidas1 > 0:
        prommed, promvar, promasi, promkurt = np.mean(MEDIA), np.mean(VAR), np.mean(ASI), np.mean(CURT)
        promsum, prom_porc = np.mean(SUMA), np.mean(PORCENTAJE)
        
        def get_kde(data):
            if len(set(data)) <= 1: return np.zeros(100)
            kde = gaussian_kde(data)
            puntos = np.linspace(min(data), max(data), 100)
            return kde(puntos)

        f1, f2, f3, f4 = get_kde(MEDIA), get_kde(VAR), get_kde(ASI), get_kde(CURT)
        f5, f10 = get_kde(SUMA), get_kde(PORCENTAJE)

        def get_mode(data, densities): 
            idx_max = np.argmax(densities)
            puntos = np.linspace(min(data), max(data), 100)
            return puntos[idx_max]
        
        m1, m2, m3, m5 = get_mode(MEDIA, f1), get_mode(VAR, f2), get_mode(ASI, f3), get_mode(SUMA, f5)
        std_m = np.sqrt(m2) if m2 > 0 else 1

      
        F1.append(f1); F2.append(f2); F3.append(f3); F4.append(f4)
        F5.append(f5); F10.append(f10)
        
        
        F6.append(m1 / std_m); F7.append(m1 / std_m) 
        F8.append(m3 / std_m); F9.append(m5 / std_m)
        F11.append(np.mean(f10) / std_m) 

        with open('Promedios.txt', 'a') as f:
            f.write(f"{prommed:.5e} {promvar:.5e} {promsum:.5e} {prom_porc:.5e}\n")

# --- final graphs ---
titulos = ['Media', 'Varianza', 'Asimetria', 'Curtosis', 'Suma', 'Ratio M1', 'Ratio M2', 'Ratio M3', 'Ratio M5', 'Porcentaje', 'Ratio P']
funciones = [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11]

for idx, func_list in enumerate(funciones):
    if not func_list: continue
    plt.figure()
    for f in func_list:
        plt.plot(f) if isinstance(f, np.ndarray) else plt.axhline(y=f, color='r', linestyle='--')
    plt.title(titulos[idx])
    plt.savefig(f'Grafica_{idx}.jpg')
    plt.close()

print(f"Tiempo total: {time.time() - start_time:.2f} s")
    

     


