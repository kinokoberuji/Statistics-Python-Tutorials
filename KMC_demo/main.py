warning_status = "ignore"
import warnings
warnings.filterwarnings(warning_status)
with warnings.catch_warnings():
    warnings.filterwarnings(warning_status, category=DeprecationWarning)

from constants import *
from data_preparation import *
from kmeans_cluster import *
from descriptive import *

def main()->None:

    df, t_df = data_generation(df_link)
    odf = df.copy()
    Elbow_plot(t_df)
    labs, centroids = kmeans_cluster(t_df, k = 2)

    describe_clusters(df = df, labs = labs, dec=3)

    plot_kde(df=odf, labs=labs)

if __name__ == '__main__':
    main()





