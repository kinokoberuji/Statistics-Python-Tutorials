
import os

df_link = 'https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Fertility.csv'

col_names = {'Age':'Tuổi', 
             'LowAFC': 'Số nang nhỏ nhất', 
             'MeanAFC': 'Số nang trung bình', 
             'FSH': 'Đỉnh FSH',
             'E2':'Mức sinh sản', 
             'MaxE2':'Mức sinh sản tối đa',
             'MaxDailyGn':'Đỉnh Gonadotropin hằng ngày',
             'TotalGn':'Gonadotropin tổng', 
             'Oocytes':'Số lượng trứng', 
             'Embryos':'Số lượng phôi',}

bicolor_pal = ['#e6154c','#0e89ed']

def show_dir(path: str):
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 3 * (level) + '|__'
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1) + '|__'
        for fname in files:
            print(f"{subindent}{fname}")

cur_path = os.getcwd()
viz_folder = os.path.join(cur_path, 'Graph')
output_folder = os.path.join(cur_path, 'Output')

if os.path.isdir(viz_folder):
    pass
else:
    os.mkdir(viz_folder)

if os.path.isdir(output_folder):
    pass
else:
    os.mkdir(output_folder)






