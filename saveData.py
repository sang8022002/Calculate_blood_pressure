# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.datasets import electrocardiogram
# from scipy.signal import find_peaks
# x = electrocardiogram()[2000:4000]
# x = electrocardiogram()[17000:18000]
# peaks, properties = find_peaks(x, prominence= 1, width=20)
# properties["prominences"], properties["widths"]
#     #(array([1.495, 2.3  ]), array([36.93773946, 39.32723577]))
# plt.plot(x)
# plt.plot(peaks, x[peaks], "x")
# plt.vlines(x=peaks, ymin=x[peaks] - properties["prominences"],ymax = x[peaks], color = "C1")
# plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],xmax=properties["right_ips"], color = "C1")
# plt.show()
# import os
# import numpy as np
# import wavio as wa
# import sounddevice as sd
# import ppfunctions_1 as ppf
# import scipy.io.wavfile as wf
#
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.datasets import electrocardiogram
# from scipy.signal import find_peaks
# x = electrocardiogram()[17000:18000]
# peaks, properties = find_peaks(x, prominence=1, width=20)
# properties["prominences"], properties["widths"]

# plt.plot(x)
# plt.plot(peaks, x[peaks], "x")
# plt.vlines(x=peaks, ymin=x[peaks] - properties["prominences"], ymax = x[peaks], color = "C1")
# plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"], xmax=properties["right_ips"], color = "C1")
# plt.show()
# diract = [1, 0, 0, 0, 0, 0, 0, 0, 0]
# import numpy as np
# my_array = np.zeros(10000, dtype=int)
# my_array[0] = 1 
import heartpy as hp
# diract_filter = hp.filter_signal(my_array, cutoff = [25, 120], sample_rate = 50,order = 3, filtertype='bandpass')
# indices = [i for i in range(len(diract_filter))]
# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(2, 1, sharex=True)
# axs[0].plot(indices, diract_filter)
# axs[0].set_xlabel("so mau")
# axs[0].set_ylabel("gia tri adc pcg")
# axs[0].set_title("PCG raw")

# axs[1].plot(indices, diract_filter)
# axs[1].set_xlabel("so mau")
# axs[1].set_ylabel("gia tri adc pcg")
# axs[1].set_title("PCG after filter")

# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal

# # Tạo một tín hiệu xung dirac
# fs = 1000  # Tần số lấy mẫu (Hz)
# t = np.arange(-0.5, 0.5, 1/fs)
# x = np.zeros_like(t)
# x[len(t)//2] = 1  # Tạo xung dirac tại giữa mảng

# # Thiết kế bộ lọc thông dải
# f1, f2 = 25, 120  # Tần số cắt dưới và trên của bộ lọc (Hz)
# b, a = signal.butter(4, [f1, f2], btype='band', fs=fs)
# x1 = np.sin(2 * np.pi * 100 * t)
# diract_filter = hp.filter_signal(x1, cutoff = [20, 50], sample_rate = fs,order = 3, filtertype='bandpass')

# # Áp dụng bộ lọc
# y = signal.lfilter(b, a, x)

# # Vẽ đồ thị
# plt.figure(figsize=(12, 6))

# plt.subplot(2, 1, 1)
# plt.title('Tín hiệu xung dirac')
# plt.plot(t, x1)
# plt.grid(True)

# plt.subplot(2, 1, 2)
# plt.title('Tín hiệu sau khi qua bộ lọc thông dải')
# plt.plot(t, diract_filter)
# plt.grid(True)

# plt.tight_layout()
# plt.show()
# import pandas as pd

# # Định nghĩa một hàm tính toán tuỳ chỉnh
# def custom_function(data):
#     return sum(data) * 2  # ví dụ, gấp đôi tổng của cửa sổ

# # Tạo một DataFrame
# df = pd.DataFrame({'values': range(10)})

# # Áp dụng hàm rolling với cửa sổ trượt có kích thước 3 và áp dụng hàm custom_function
# df['custom_roll'] = df['values'].rolling(window=3).apply(custom_function, raw=True)

# print(df)
# import numpy as np
# from scipy.stats import mannwhitneyu

# def calculate_mann_whitney_p_value(group1, group2):
#     # Sử dụng hàm mannwhitneyu để tính giá trị U và p
#     u_statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
#     return u_statistic, p_value

# # Dữ liệu mẫu
# group1 = np.array([7, 5, 3, 9])
# group2 = np.array([1, 2, 8, 6, 4])

# # Tính và in giá trị U và p
# u_statistic, p_value = calculate_mann_whitney_p_value(group1, group2)
# print("Giá trị U:", u_statistic)
# print("Giá trị p:", p_value)
# pip install numpy
# import numpy as np
# def find_closest_values(arr, target):
#     # Tính khoảng cách giữa mỗi giá trị trong mảng và giá trị mục tiêu
#     distances = [(abs(value - target), value) for value in arr]

#     # Sắp xếp các cặp (khoảng cách, giá trị) theo khoảng cách
#     sorted_distances = sorted(distances)

#     # Lấy ra hai giá trị có khoảng cách nhỏ nhất
#     return sorted_distances[0][1], sorted_distances[1][1]

# # Ví dụ sử dụng
# arr = [1, 5, 9, 15, 20, 25]
# target = 10
# closest_values = find_closest_values(arr, target)
# print("Hai giá trị gần nhất với", target, "là:", closest_values)
def split_and_extract_middle(data):
    # Chia mảng thành 3 phần gần bằng nhau
    n = len(data)
    third = n // 3
    start_index = third
    end_index = n - third if n % 3 == 0 else n - third + 1

    # Trả về phần giữa của mảng
    return data[start_index:end_index]

# Mẫu mảng data
data_example = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Áp dụng hàm và in ra phần giữa
middle_part = split_and_extract_middle(data_example)
print(middle_part)