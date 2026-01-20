from scipy.stats import kendalltau
import numpy as np

rank_similarity_coefficient = lambda x, y: kendalltau(x, y)[0]
weighted_rank_similarity_coefficient = lambda x, y: np.sum(np.subtract(y, x)/(np.log2(x + np.finfo(float).eps +2)))

exposure_diff = lambda x, y: np.sum(np.abs(1/(np.log2(np.array(x) + 2)) - 1/(np.log2(np.array(y) + 2))))

array1 = [1,2,3,4,5]

array2 = [5,4,3,2,1]

print("different list" )
print(rank_similarity_coefficient(array1, array2))

print(weighted_rank_similarity_coefficient(array1, array2))

print(exposure_diff(array1, array2))
print("similar list" )
print(rank_similarity_coefficient(array1, array1))

print(weighted_rank_similarity_coefficient(array1, array1))

print(exposure_diff(array1, array1))
array1 = [[[1,2,3,4,5],
          [1,2,3,4,5],
          [1,2,3,4,5]]]*2
print(np.asarray(array1).transpose((0, 2, 1)))

print(np.mean(np.asarray(array1).transpose((0, 2, 1)), axis=2))

print(["{}@{}".format("approach",str(i)) for i in [1,3,5,7,10] ])

import pathlib
filepath = pathlib.Path("RankingShap/results")
# print(filepath.resolve())
# print(np.genfromtxt(filepath / "run{} MQ.csv".format(1), delimiter=","))

import pandas as pd

n_runs = 2
runs = []
dataset = 'MQ2008'

for i in range(n_runs):
    runs.append(pd.read_csv(filepath / "run{} {}.csv".format(i+1, dataset)))

total = pd.concat(runs)
mean = total.groupby("approach").mean()

mean = mean.reset_index()

mean[['approach', 'at']] = mean.approach.str.split("@", expand = True)


metrics = [ "Pre_ken", "Del_ken", "Pre_exp", "Del_exp"]
proper_names = ["Preservation kendalltau", "Deletion kendalltau", \
             "Preservation exposure difference", "Deletion exposure difference"]

metric_dfs = []

colours = {'greedy_iter': '#ffd166', 'random': '#a8d48a', 'rankingshapK': '#d6335c', 'rankingshapW': '#993399', 'greedy_iter_full': '#222222', 'pointwise_lime': '#99d6c2', 'pointwise_shap': '#bab2f7', 'rankinglime': '#f0b2c2'}

for metric in metrics:
    metric_dfs.append(mean[["approach", 'at', metric]])


    for k in [1,3,5,7,10]:
        metric_dfs[-1][str(k)] = metric_dfs[-1].loc[metric_dfs[-1]['at'] == str(k), [metric]]


    metric_dfs[-1] = metric_dfs[-1].drop(['at', metric], axis= 1)

    metric_dfs[-1] = metric_dfs[-1].groupby('approach').sum()
    
# for metric_df in metric_dfs:.
i = 0
for i in range(4):
    if i%2 == 1:
        metric_dfs[i] = metric_dfs[i] * -1
    plot = metric_dfs[i].T.plot(linewidth=2, color = [colours.get(name, '#333333') for name in metric_dfs[i].T.columns])
    plot = plot.set_title(proper_names[i] + " " + dataset)


    plot.figure.savefig("{}.jpg".format(metrics[i]))
