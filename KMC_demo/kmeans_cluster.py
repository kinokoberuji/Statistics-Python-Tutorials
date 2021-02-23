import os
import pandas as pd
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (8,6)
plt.rcParams.update({'font.size': 10})

from constants import *

def Elbow_plot(t_df:None) -> None:
    '''Vẽ biểu đồ Elbow để xác định số phân cụm tối ưu
    '''
    wcss = []

    for i in range(1,7):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=500, n_init=10, random_state=123)
        kmeans.fit(t_df)
        wcss.append(kmeans.inertia_)

    plt.rcParams["figure.figsize"] = (8,6)
    plt.rcParams.update({'font.size': 14})

    plt.plot(range(1, 7), wcss)
    plt.title('Số phân cụm tối ưu = 2')
    plt.vlines(2,ymin=min(wcss),ymax=max(wcss),linestyles='dashed',color='k',alpha = 0.5)
    plt.hlines(wcss[1],xmin= 1,xmax=6,linestyles='dashed',color='k',alpha = 0.5)
    plt.scatter(2,wcss[1],s=50)
    plt.xlabel('Số phân cụm')
    plt.ylabel('Within-Cluster-Sum-of-Squares')
    plt.xticks(range(1,7))

    fig_name = os.path.join(viz_folder, f"Elbow_plot.svg")
    plt.savefig(fname = fig_name, format = 'svg', dpi = 300)

    plt.show()

def kmeans_cluster(t_df: pd.DataFrame, k = 2):
    '''Phân cụm dataframe
    :t_df: dataframe đã chuẩn hóa,
    :k: số phân cụm mong muốn

    :Output: 2 np.array: chuỗi labels và centroids
    '''
    kmeans = KMeans(n_clusters=k, 
                    init='k-means++', 
                    max_iter=300, 
                    n_init=10, 
                    random_state=123)

    pred_y = kmeans.fit_predict(t_df)
    centroids = kmeans.cluster_centers_

    print('Đã thực hiện phân cụm')

    return pred_y, centroids


    


