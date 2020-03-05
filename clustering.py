# coding=utf-8
from sklearn.cluster import AgglomerativeClustering #Hierarchical
from sklearn.cluster import KMeans #KMeans;
from sklearn.mixture import GaussianMixture #EM
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd


# Lêr dataframe
df = pd.read_csv('csv/Diabetes.csv',encoding='utf-8')

# Remove última coluna com a classe
labels = df['class'].values
X = df.drop(columns=['class'])

print('\n1. Executar os três algoritmos de clustering sobre o dataset, variando o número de grupos (k) de 2 a 3')

models = [
    { 't': "AgglomerativeClustering (k = 2)", 'm': AgglomerativeClustering(n_clusters=2) },
    { 't': "AgglomerativeClustering (k = 3)", 'm': AgglomerativeClustering(n_clusters=3) },
    { 't': "KMeans (k = 2)", 'm': KMeans(n_clusters=2) },
    { 't': "KMeans (k = 3)", 'm': KMeans(n_clusters=3) },
    { 't': "GaussianMixture (k = 2)", 'm': GaussianMixture(n_components=2) },
    { 't': "GaussianMixture (k = 3)", 'm': GaussianMixture(n_components=3) }
]

print('\n2. Visualizar os resultados através de gráficos')

for m in models:
    predicts = m['m'].fit_predict(X)
    t = m['t']

    X_PCA = PCA(n_components=2).fit_transform(X)

    plt.scatter(X_PCA[:, 0], X_PCA[:, 1], c=predicts, s=50, cmap='rainbow',alpha=0.5)

    plt.title(t)
   
    plt.savefig(t+'.png')
    plt.clf()


print('\n3. Fazer a validação dos grupos através da utilização dos índices (Davies Bouldin e Silhouette)')

print('{}\t{}\t{}'.format('Model', 'Bouldin', 'Silhouette'))
for m in models:
    labels = m['m'].fit_predict(X)
    dbScore = davies_bouldin_score(X, labels)
    sScore = silhouette_score(X, labels, metric='euclidean')
    print('{}\t{}\t{}'.format(m['t'], dbScore, sScore))

print('\n4. Mostrar os resultados de ambos os índices para os agrupamentos criados')

print(
"""
--------------------------------------------------------------------------
Modelo	                            Bouldin	            Silhouette
--------------------------------------------------------------------------
AgglomerativeClustering (k = 2)	    0.7330018210488929	0.5532678504628996
KMeans (k = 2)	                    0.7133822795826191	0.5687897205830247
GaussianMixture (k = 2)	            0.8604650094596924	0.3919496047300402

AgglomerativeClustering (k = 3)	    0.6041813066360704	0.5281675826566276
KMeans (k = 3)	                    0.6680978941351275	0.5104287492214447
GaussianMixture (k = 3)	            0.7379219639790152	0.4240572985480604
--------------------------------------------------------------------------
"""
)

print('\n5. Analisar os resultados obtidos')

print("""
Quanto maior o silhouette, melhor é o resultado. Já o DB, é o contrário. 
Dessa forma para k=2, o KMeans obteve melhor resultado e o GaussianMixture o pior.
Já para k = 3, o AgglomerativeClustering foi o melhor e o GaussianMixture permaneceu como a pior solução.
"""
)
