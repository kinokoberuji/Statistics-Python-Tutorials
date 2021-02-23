import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
import pingouin as pg
from constants import *

def describe_clusters(df:pd.DataFrame, labs: np.array, dec:int)->None:
    '''Lập bảng so sánh 10 biến giữa 2 phân cụm
    :df: dataframe dữ liệu gốc,
    :labs: chuỗi labels cho 2 phân cụm
    :dec: số lẻ cho việc làm tròn

    Kết quả là 1 dataframe bảng thống kê mô tả và kiểm định t, 
    sao lưu trong thư mục Output
    ''' 

    df['C'] = labs

    res_df = df.groupby('C').agg(lambda x: f"{np.round(np.mean(x),dec)} ± {np.round(np.std(x),dec)}").T
    res_df.columns = ['Cụm 1', 'Cụm 2']

    res_df['Toàn thể'] = df.agg(lambda x: f"{np.round(np.mean(x),dec)} ± {np.round(np.std(x),dec)}").T

    res_df = res_df[['Toàn thể', 'Cụm 1', 'Cụm 2']]

    p_val = []

    for v in res_df.index:
        p = pg.ttest(df[df['C'] == 0][v], df[df['C']==1][v], tail='one-sided')['p-val'][0]
        p_val.append(p)

    res_df['Giá trị p'] = p_val

    res_df.index = res_df.index.map(col_names)

    print('Kết quả thống kê mô tả 2 phân cụm:')
    print('='*30)
    print(res_df.to_string())

    csv_name = os.path.join(output_folder, f"Table.xlsx")

    res_df.to_excel(csv_name, index = True, encoding='utf-8')
    
def plot_kde(df: pd.DataFrame, labs: np.array)->None:
    '''Vẽ pairplot khảo sát tương quan và phân bố của 10 biến
    :df: dataframe gốc
    :labs: chuỗi labels phân cụm
    '''
    df.columns = df.columns.map(col_names)
    df['C'] = labs
    sns.set_palette(sns.color_palette(bicolor_pal))

    sns.pairplot(data = df, 
             kind = 'kde', 
             diag_kind='hist',
             hue = 'C',
             plot_kws={'shade':True, 'alpha': 0.3}
             )
    
    fig_name = os.path.join(viz_folder, f"KDE_plot.svg")
    plt.savefig(fname = fig_name, format = 'svg', dpi = 300)


