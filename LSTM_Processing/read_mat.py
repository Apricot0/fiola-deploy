# # #!/usr/bin/env python
# # # -*- coding: utf-8 -*-
# # """
# # 此脚本用于打开一个 .mat 文件并读取其中的所有内容。

# # 使用方法:
# #     python read_mat.py 文件路径.mat
# # """

# # import sys
# # import scipy.io

# # def read_mat_file(filepath):
# #     """
# #     读取 .mat 文件，并返回一个字典，其键为变量名，值为对应数据。
# #     """
# #     try:
# #         data = scipy.io.loadmat(filepath)
# #         return data
# #     except Exception as e:
# #         print(f"读取 {filepath} 时发生错误: {e}")
# #         sys.exit(1)

# # def main():
# #     if len(sys.argv) != 2:
# #         print("用法: python read_mat.py 文件路径.mat")
# #         sys.exit(1)
    
# #     filepath = sys.argv[1]
# #     mat_data = read_mat_file(filepath)
    
# #     print("读取到的数据:")
# #     for key, value in mat_data.items():
# #         # 注意：MAT 文件中通常会包含 '__header__'、'__version__'、'__globals__' 等信息
# #         print(f"键: {key}")
# #         print("值:")
# #         print(value)
# #         print("-" * 40)

# # if __name__ == "__main__":
# #     main()
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 此脚本用于读取指定目录下的 6 个 .mat 文件，
# 并输出每个文件中 arena_x、arena_y、spks、t 变量的数据形状或长度。

# 用法:
#     python3 read_mat_files.py [mat文件所在目录]

# 如果不指定目录，则默认读取当前目录下的 .mat 文件。
# """

# import os
# import sys
# import scipy.io

# def read_mat_file(filepath):
#     """
#     读取 .mat 文件，并返回一个字典，其键为变量名，值为对应数据。
#     """
#     try:
#         data = scipy.io.loadmat(filepath)
#         return data
#     except Exception as e:
#         print(f"读取 {filepath} 时发生错误: {e}")
#         sys.exit(1)

# def process_mat_file(filepath):
#     """
#     读取指定的 .mat 文件，并输出 arena_x、arena_y、spks、t 的数据形状或长度。
#     """
#     data = read_mat_file(filepath)
#     # 我们关心的键列表
#     keys = ['arena_x', 'arena_y', 'spks', 't']
#     print(f"文件: {os.path.basename(filepath)}")
#     for key in keys:
#         if key in data:
#             value = data[key]
#             # 过滤 MATLAB 中自带的多余信息（如 '__header__' 等）
#             if hasattr(value, 'shape'):
#                 shape = value.shape
#                 print(f"  {key} 的 shape: {shape}")
#                 if len(shape) == 1:
#                     print(f"  {key} 的长度: {len(value)}")
#                 elif len(shape) >= 1:
#                     # 这里认为第一维为样本数
#                     print(f"  {key} 的长度（第一维）: {shape[0]}")
#             else:
#                 print(f"  {key} 的值: {value}")
#         else:
#             print(f"  未找到键: {key}")
#     print("-" * 40)

# def main():
#     # 如果传入参数，则第一个参数作为目录，否则默认为当前目录
#     if len(sys.argv) >= 2:
#         directory = sys.argv[1]
#     else:
#         directory = "."
    
#     if not os.path.isdir(directory):
#         print(f"错误: {directory} 不是有效的目录！")
#         sys.exit(1)
    
#     # 找到目录下所有 .mat 文件
#     all_files = [f for f in os.listdir(directory) if f.endswith('.mat')]
#     if not all_files:
#         print(f"目录 {directory} 中没有找到 .mat 文件！")
#         sys.exit(0)
    
#     # 只处理前 6 个文件（如果文件数不足 6，则处理全部）
#     files_to_process = all_files[:6]
#     print(f"将在目录 {directory} 中处理 {len(files_to_process)} 个 .mat 文件。")
#     print("=" * 40)
    
#     for f in files_to_process:
#         filepath = os.path.join(directory, f)
#         process_mat_file(filepath)

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
此脚本用于读取指定目录下最多 6 个 .mat 文件，
分别读取其中的 arena_x, arena_y, spks, t 四个变量，
输出各变量的形状（或长度），并将这些变量合并成一个特征矩阵，
然后输出合并后矩阵的 shape。

合并方法：
  假设 arena_x, arena_y, t 为 (N, 1) 形式，spks 为 (channels, N) 形式，
  则对 spks 进行转置（得到 (N, channels)），再按列拼接：
      combined = [arena_x, arena_y, spks.T, t]
  这样，每个样本的特征维度为： 1 + 1 + (channels) + 1 = channels + 3

注意：若各变量的样本数（N）不一致，将自动截断到最小样本数。
"""

import os
import sys
import scipy.io
import numpy as np

def read_mat_file(filepath):
    """
    读取 .mat 文件，并返回一个字典，其键为变量名，值为对应数据。
    """
    try:
        data = scipy.io.loadmat(filepath)
        return data
    except Exception as e:
        print(f"读取 {filepath} 时发生错误: {e}")
        sys.exit(1)

def process_and_combine(filepath):
    """
    读取指定的 .mat 文件，输出 arena_x, arena_y, spks, t 的形状，
    并将它们合并成一个特征矩阵。
    """
    data = read_mat_file(filepath)
    keys = ['arena_x', 'arena_y', 'spks', 't']
    variables = {}
    
    # 依次获取变量，保证至少为二维数组
    for key in keys:
        if key in data:
            value = np.array(data[key])
            value = np.squeeze(value)
            if value.ndim == 1:
                value = value.reshape(-1, 1)
            variables[key] = value
        else:
            print(f"文件 {os.path.basename(filepath)} 中未找到关键变量：{key}")
            return None

    # 输出各变量的形状
    print(f"文件: {os.path.basename(filepath)}")
    for key in keys:
        shape = variables[key].shape
        print(f"  {key} 的 shape: {shape}")
        if len(shape) == 1:
            print(f"  {key} 的长度: {len(variables[key])}")
        elif len(shape) >= 1:
            print(f"  {key} 的长度（第一维）: {shape[0]}")
    
    # 对 spks，确保是二维数组：一般应为 (channels, samples)
    if variables['spks'].ndim == 1:
        variables['spks'] = variables['spks'].reshape(1, -1)
    
    # 获取各变量样本数（第一维），spks 的样本数为第二维
    N_x = variables['arena_x'].shape[0]
    N_y = variables['arena_y'].shape[0]
    N_t = variables['t'].shape[0]
    N_spks = variables['spks'].shape[1]
    N_min = min(N_x, N_y, N_t, N_spks)
    
    if N_min < N_x or N_min < N_y or N_min < N_t or N_min < N_spks:
        print(f"  注意：各变量样本数不一致，将自动截断到最小样本数：{N_min}")
    
    # 截断各变量
    arena_x = variables['arena_x'][:N_min]
    arena_y = variables['arena_y'][:N_min]
    t = variables['t'][:N_min]
    spks = variables['spks'][:, :N_min]  # spks: (channels, N_min)
    
    # 合并：对 spks 进行转置得到 (N_min, channels)
    spks_T = spks.T
    combined = np.concatenate([arena_x, arena_y, spks_T, t], axis=1)
    print(f"  合并后的特征矩阵的 shape: {combined.shape}")
    return combined

def main():
    # 如果传入参数，则第一个参数作为目录，否则默认当前目录
    if len(sys.argv) >= 2:
        directory = sys.argv[1]
    else:
        directory = "."
    
    if not os.path.isdir(directory):
        print(f"错误: {directory} 不是有效的目录！")
        sys.exit(1)
    
    # 找到目录下所有 .mat 文件
    all_files = [f for f in os.listdir(directory) if f.endswith('.mat')]
    if not all_files:
        print(f"目录 {directory} 中没有找到 .mat 文件！")
        sys.exit(0)
    
    # 只处理前 6 个文件（若不足6个则全部处理）
    files_to_process = all_files[:6]
    print(f"将在目录 {directory} 中处理 {len(files_to_process)} 个 .mat 文件。")
    print("=" * 40)
    
    for f in files_to_process:
        filepath = os.path.join(directory, f)
        combined = process_and_combine(filepath)
        print("-" * 40)

if __name__ == "__main__":
    main()
