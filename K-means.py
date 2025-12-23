from sklearn.cluster import KMeans, 
import pandas as pd

file = r"C:\data.csv"

DF_IH = pd.read_csv(file)

Features = DF_IH[["Pixels > Noise"]]
Features = DF_IH[["Pixels > Noise","Mode","Standard Deviation"]]
#Features = DF_IH.drop(columns=["Label","Muestra"])

# ---------------------------- K-Means ---------------------------- #
Kmeans = KMeans(n_clusters=2, init='k-means++', n_init=20, max_iter=500, tol=1e-4)
DF_IH['flag'] = Kmeans.fit_predict(Features)

matching_rows = (DF_IH['flag'] == DF_IH['Label']).sum()
percentage_match = (matching_rows / len(DF_IH)) * 100
print(f"Percentage of rows with matching 'KMeans flag' and 'label': {percentage_match:.2f}%")
