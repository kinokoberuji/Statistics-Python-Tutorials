import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from constants import *

def data_generation(df_link: str):
    '''Hàm tải data và chuẩn hóa thang đo
    :df_link: link đến file csv (str)
    :output: 2 dataframes (gốc và hoán chuyển min/max)
    '''
    df = pd.read_csv(df_link,sep = ',', index_col=0)
    scaler = MinMaxScaler()
    
    t_df = df.copy()
    t_df.loc[:] = scaler.fit_transform(df)

    print("Thực hiện tải dữ liệu và hoán chuyển Min-Max")

    return df, t_df

