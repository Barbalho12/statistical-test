# coding=utf-8
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp #pip3 install scikit-posthocs
import pandas as pd

df = pd.read_csv('csv/comparativo_tecnicas.csv',encoding='utf-8')

print('\nFriedman')

stat, p = friedmanchisquare(df['dt'].tolist(), df['nb'].tolist(), df['mlp'].tolist())
print('p=%.3f' % (p))
if p > 0.05:
    print('Não há diferença significativa')

else:
    print('Há diferença significativa')

    print('\n Posthoc')
    posthoc = sp.posthoc_nemenyi([df['dt'].tolist(), df['nb'].tolist(), df['mlp'].tolist()])
    print(posthoc)



