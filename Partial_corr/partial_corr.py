# Source code cho bài thực hành phân tích tương quan bộ phận

import numpy as np
import pandas as pd
from pandas_flavor import register_dataframe_method
from scipy.stats import pearsonr, spearmanr, kendalltau

import seaborn as sns
import matplotlib.pyplot as plt

from typing import List

def flatten(nested_lst):
    '''Hàm chuyển một list hỗn hợp chứa cả
    string, list, None thành 1 list duy nhất
    '''
    for i in nested_lst:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            if isinstance(i,str):
                yield i
            else:
                continue

def _cor(xs: pd.Series, ys:pd.Series, method = 'pearson'):
    '''Hàm phân tích tương quan giữa 2 biến số xs, ys
    :xs: panda Series
    :ys: panda Series
    Lưu ý: kích thước phải tương đồng
    :method: tên phương pháp, hỗ trợ Pearson, Spearman và Kendall

    :Return:
    :r: Hệ số tương quan Pearson's r, Spearman's Rho...
    :p_val: giá trị p 
    '''
    if method == 'pearson':
        r,p_val = pearsonr(xs, ys)
    elif method == 'spearman':
        r,p_val = spearmanr(xs, ys)
    elif method == 'kendall':
        r,p_val = kendalltau(xs, ys)
    else:
        raise ValueError(f"Không nhận ra {method}, chỉ hỗ trợ method spearman, pearson hoặc kendalltau")

    return r, p_val

# Hàm phân tích tương quan bộ phận

@register_dataframe_method
def partial_correlation(data: pd.DataFrame, 
                        x: str=None, y: str=None, 
                        covar: List[str]=None, 
                        x_covar: List[str]=None, 
                        y_covar: List[str]=None,
                        method = 'pearson', 
                        graph = True):

    '''Hàm phân tích tương quan bộ phận
    :data: 1 dataframe 
    :x: tên biến x (str hoặc list chứa 1 str)
    :y: tên biến y (str hoặc list chứa 1 str)
    :covar: danh sách hiệp biến của cả x và y (partial corr), hoặc None
    :x_covar: danh sách hiệp biến của riêng x (semi-partial corr), hoặc None
    :y_covar: danh sách hiệp biến của riêng y (semi-partial corr), hoặc None

    Lưu ý: không thể áp dụng đồng thời cả 2: covar và 1 trong x_covar/y_covar, 
    tuy nhiên có thể chỉ cần x_covar hoặc y_covar, hoặc cả 2

    :method: tên phương pháp, hỗ trợ Pearson, Spearman và Kendall
    :graph: có vẽ biểu đồ hay không

    :return: 1 dataframe chứa kết quả hệ số tương quan 
    và p_value của 2 phân tích tương quan bình thường x,y và bộ phận (x,y,covar)

    :Cách sử dụng:
    Hàm này có thể sử dụng rời với đối số data là pd.DataFrame, 
    hoặc như 1 method của pandas DataFrame, khi đó không cần đối số data
    '''
    
    # Kiểm tra tính hợp lệ của arguments
    
    assert isinstance(covar, (str, list, type(None))), 'Lỗi: x_covar phải là list hoặc string'
    assert isinstance(x_covar, (str, list, type(None))), 'Lỗi: x_covar phải là list hoặc string'
    assert isinstance(y_covar, (str, list, type(None))), 'Lỗi: y_covar phải là list hoặc string'
    
    if (covar is not None) & ((x_covar is not None) | (y_covar is not None)):
        raise ValueError('Lỗi: Không thể áp dụng đồng thời cả 2: covar và x_covar hoặc y_covar')
    if covar is not None:
        assert x not in covar, f"Lỗi: tập hợp covar không thể bao gồm biến {x}"
        assert y not in covar, f"Lỗi: tập hợp covar không thể bao gồm biến {y}"
    if x_covar is not None:
        assert x not in x_covar, f"Lỗi: tập hợp x_covar không thể bao gồm biến {x}"
        assert y not in x_covar, f"Lỗi: tập hợp x_covar không thể bao gồm biến {y}"
    if y_covar is not None:
        assert x not in y_covar, f"Lỗi: tập hợp y_covar không thể bao gồm biến {x}"
        assert y not in y_covar, f"Lỗi: tập hợp y_covar không thể bao gồm biến {y}"
    
    # Kiểm tra tính hợp lệ của dữ liệu
    col = set(i for i in flatten([x, y, covar, x_covar, y_covar]))
    outsider = [c for c in col if not c in data]
    
    assert len(outsider) == 0, f"Không tìm thấy biến {','.join(c for c in outsider)} trong dataframe"
    assert all([data[c].dtype.kind in 'fi' for c in col]), 'Lỗi: Chỉ chấp nhận biến số float hoặc int'
    
    data = data[col].dropna(how = 'any', axis = 0)
    assert data.shape[0] >= 5, 'Dữ liệu cần có ít nhất 5 trường hợp'
    
    std_df = (data[col] - data[col].mean(axis = 0))/data[col].std(axis = 0)
    
    # Chuyển covars đơn (string) thành list 
    if isinstance(covar, str):
        covar = [covar]
    if isinstance(x_covar, str):
        x_covar = [x_covar]
    if isinstance(y_covar, str):
        y_covar = [y_covar]
        
    if covar is not None:
        # Phân tích tương quan bộ phận

        cov_mat = np.atleast_2d(std_df[covar])
        W_x, W_y = np.linalg.lstsq(cov_mat, std_df[x], rcond=None)[0], np.linalg.lstsq(cov_mat, std_df[y], rcond=None)[0]
        resid_x, resid_y = std_df[x] - cov_mat@W_x, std_df[y] - cov_mat@W_y
        
    else:
        # Phân tích tương quan bán bộ phận
        resid_x, resid_y = std_df[x], std_df[y]

        if x_covar is not None:
            cov_mat = np.atleast_2d(std_df[x_covar])
            W_x = np.linalg.lstsq(cov_mat, std_df[x], rcond=None)[0]
            resid_x = std_df[x] - cov_mat@W_x
        if y_covar is not None:
            cov_mat = np.atleast_2d(std_df[y_covar])
            W_y = np.linalg.lstsq(cov_mat, std_df[y], rcond=None)[0]
            resid_y = std_df[y] - cov_mat@W_y
        
    pr,pp_val = _cor(xs = resid_x, ys = resid_y, method = method)
    
    # Tương quan toàn thể
    r,p_val = _cor(xs = std_df[x], ys = std_df[y], method = method)
    
    # Nếu vẽ biểu đồ
    if graph:
        fig, axes = plt.subplots(1,2, figsize = (8,3.5))
        
        sns.regplot(x = resid_x, 
                    y = resid_y, 
                    ax = axes[0], 
                    scatter_kws = {'alpha':0.1,},
                    line_kws = {'alpha':0.8,},
                    color = '#d10d55',
                    label = 'Partial corr')
        
        axes[0].legend()
        axes[0].set_xlabel(f"std_residual {x}")
        axes[0].set_ylabel(f"std_residual {y}")

        sns.regplot(data = data, 
                    x=x, y=y, 
                    ax = axes[1], 
                    scatter_kws = {'alpha':0.1,},
                    line_kws = {'alpha':0.8,},
                    color = '#1296de',
                    label = 'Correlation')
        
        axes[1].legend()
        axes[1].set_xlabel(x)
        axes[1].set_ylabel(y)
        plt.show()
    
    return pd.DataFrame({f"Hệ số tương quan {method}":[r, pr], 'Giá trị p':[p_val, pp_val]}, index = ['Toàn phần', 'Bộ phận']).T