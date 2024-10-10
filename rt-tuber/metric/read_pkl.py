'''读取pkl文件'''

import pickle
import os

def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f,encoding='iso-8859-1')
    return data

def write_pkl(data, file_path): 
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    

if __name__ == '__main__':
    file_path = 'data/ucf24-101/UCF101v2-GT.pkl'

    output_path = 'data/ucf24-101/UCF101v2_newlabel-GT.pkl'
    data = read_pkl(file_path)
    gttubes = data['gttubes']

    new_gttubes = {}
    # 将gttubes中的数据打印出来的标签的数字都加一
    for key in gttubes:
        for key2 in gttubes[key]:
            # 将key2 +1存在new_gttubes中
            new_gttubes[key] = {key2+1: gttubes[key][key2]}

    data['gttubes'] = new_gttubes
    write_pkl(data, output_path)