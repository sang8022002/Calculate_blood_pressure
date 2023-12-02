import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
with open('ppg_pcg//93 66 87 trang.txt', 'r') as file:
    # Đọc nội dung từ tệp
    content = file.read()

# Loại bỏ tất cả các dấu cách và thay thế chúng bằng dấu phẩy
content_without_spaces = content.replace("\t", ",")

# In nội dung sau khi xử lý
#print(content_without_spaces)

# Tạo hoặc mở tệp mới để lưu nội dung đã xử lý
with open('PPG_PCG_file.csv', 'w') as new_file:
    # Ghi nội dung đã xử lý vào tệp mới
    new_file.write(content_without_spaces)


file_name = 'PPG_PCG_file.csv'  # Thay 'your_csv_file.csv' bằng tên thực tế của tệp
import csv
# Mở tệp CSV để đọc
ppg_data = []
pcg_data = []
fs = 500
windowsize = int(fs* 0.1)
with open(file_name, mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)

    # Duyệt qua từng dòng trong tệp CSV
    for row in csv_reader:
        if len(row) >= 9:  # Đảm bảo có ít nhất 7 cột trong mỗi dòng
            column_7_data = float(row[7])
            column_6_data = float(row[6])# Lấy dữ liệu từ cột thứ 7 (0-based index)
            ppg_data.append(column_7_data)
            pcg_data.append(column_6_data)
            # Sử dụng dữ liệu từ cột thứ 7 ở đây, ví dụ:
            #print(column_data)
import matplotlib.pyplot as plt

def split_and_extract_middle(data):
    # Chia mảng thành 3 phần gần bằng nhau
    n = len(data)
    third = n // 3
    start_index = third
    end_index = n - third if n % 3 == 0 else n - third + 1

    # Trả về phần giữa của mảng
    return data[start_index:end_index]

# Mẫu mảng data
# Áp dụng hàm và in ra phần giữa
ppg_data = split_and_extract_middle(ppg_data)
pcg_data = split_and_extract_middle(pcg_data)
# Dữ liệu mảng (ví dụ: danh sách các số nguyên từ 1 đến 10)
#data = [1, 2, 4, 6, 9, 11, 13, 14, 15, 16]
indices = [i for i in range(len(ppg_data))]

ch = 1
# fig, axs = plt.subplots(2, 1, sharex=True)

# axs[0].plot(indices, ppg_data)
# axs[0].set_xlabel("So mau")
# axs[0].set_ylabel("Gia tri ppg raw")
# axs[0].set_title("ppg data")

# axs[1].plot(indices, pcg_data)
# axs[1].set_xlabel("So mau")
# axs[1].set_ylabel("Gia tri pcg raw")
# axs[1].set_title("pcg data")

# plt.show()

def movmean_data(A, k):
    x = A.rolling(k,min_periods= 1, center= True).mean().to_numpy()
    return x
def movmedian_data(A, k):
    x = A.rolling(k, min_periods= 1, center= True).median().to_numpy()
    return x
ppg_data_copy = ppg_data.copy()
ppg_data_copy_frame = pd.DataFrame(ppg_data_copy)

median_data = movmedian_data(ppg_data_copy_frame, windowsize)
#median_data = median_data_frame.flatten()
median_data_frame = pd.DataFrame(median_data)
baseline_data_frame = movmean_data(median_data_frame, fs)
baseline_data = baseline_data_frame.flatten()
median_data = median_data.flatten()
ac_ppg_data = median_data - baseline_data

# plt.figure("ppg and baseline")
# plt.plot(indices, median_data)
# plt.plot(indices, baseline_data)
# plt.title("ppg and baseline")
# plt.xlabel("sampling")
# plt.ylabel("adc value ppg")
# plt.show()

# plt.figure("ac ppg")
# plt.plot(indices, ac_ppg_data)
# plt.title("ac ppg")
# plt.xlabel("sampling")
# plt.ylabel("adc value ppg")
# plt.show()

ampl, __ = find_peaks(median_data, distance=int(0.34 * fs), prominence = 16000)  #chỉnh PPG tại đây

# plt.figure("find peak ppg")
# plt.plot(indices, median_data)
# for value in ampl:
#     plt.plot(value, median_data[value], "r*")
# plt.title("find peak ppg")
# plt.xlabel("sampling")
# plt.ylabel("adc value")
# plt.show()

pcg_data_frame = pd.DataFrame(pcg_data)
pcg_median_data_frame = movmedian_data(pcg_data_frame, windowsize)
pcg_median_data = pcg_median_data_frame.flatten()
ampl1, __= find_peaks(pcg_median_data, distance=int(0.15 * fs), height=(300000,5000000))

# find peak pcg data
import heartpy as hp
pcg_filtered = hp.filter_signal(pcg_data, cutoff = [25, 120], sample_rate = fs,order = 4, filtertype='bandpass')
indices_pcg_data_filtered = [i for i in range(len(pcg_filtered))]
#pcg_filtered = -pcg_filtered
ampl_pcg_data_filtered, __= find_peaks(pcg_filtered, distance=int(0.15 * fs),prominence = 700000)# chỉnh PCG tại đây
indices_ampl_pcg_data_filtered = [i for i in range(len(ampl_pcg_data_filtered))]

ppg_filtered = hp.filter_signal(ppg_data, cutoff = [2, 200], sample_rate = fs,order = 4, filtertype='bandpass')
indices_ppg_data_filtered = [i for i in range(len(ppg_filtered))]
ampl_ppg_data_filtered, __= find_peaks(ppg_filtered, distance = int(0.34 * fs))
indices_ampl_ppg_data_filtered = [i for i in range(len(ampl_ppg_data_filtered))]


##############################################################
# fig, axs = plt.subplots(2, 1, sharex=True)
# axs[0].plot(indices_pcg_data_filtered, pcg_data)
# axs[0].set_xlabel("so mau")
# axs[0].set_ylabel("gia tri adc pcg")
# axs[0].set_title("PCG raw")

# axs[1].plot(indices_pcg_data_filtered, pcg_filtered)
# axs[1].set_xlabel("so mau")
# axs[1].set_ylabel("gia tri adc pcg")
# axs[1].set_title("PCG after filter")

# plt.show()

RR = ampl[1:] - ampl[:-1]
FHR = 60 * fs / RR  # ví dụ, gấp đôi tổng của cửa sổ
FHR_average = np.mean(FHR)
# print(f'heart rate {FHR_average}')
##################################################################
# # Sử dụng FFT để chuyển đổi sang miền tần số
# frequency_data = np.fft.fft(median_data)
# frequencies = np.fft.fftfreq(len(median_data), d=1/fs)

# # Lấy mô-đun của kết quả FFT để có được phổ công suất
# power_spectrum = np.abs(frequency_data)**2

# # Vẽ biểu đồ phổ công suất
# plt.figure()
# plt.plot(frequencies, power_spectrum)
# plt.xlim(-20, 20)
# plt.title(f'Power Spectrum {FHR_average}')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power')
# plt.grid()
# plt.show()
##################################################################

indices_ampl = [i for i in range(len(ampl))]
indices_ampl1 = [i for i in range(len(ampl1))]
# plt.figure("ppg-pcg find peak")
# fig, axs = plt.subplots(2, 1, sharex=True)
# axs[0].plot(indices, median_data)
# axs[0].set_xlabel("so mau")
# axs[0].set_ylabel("gia tri adc ppg")
# axs[0].set_title(f'PPG after filter, sum peak = {len(ampl)}')
# for value in ampl:
#     axs[0].plot(value, median_data[value],"r*")

# axs[1].plot(indices_pcg_data_filtered, pcg_filtered)
# axs[1].set_xlabel("so mau")
# axs[1].set_ylabel("gia tri adc pcg")
# axs[1].set_title(f'PCG after filter, sum peak = {len(indices_ampl_pcg_data_filtered)}')
# for value in ampl_pcg_data_filtered:
#     axs[1].plot(value, pcg_filtered[value], "r*")
# plt.show()

VTT = []

# print(len(ampl))
# print(len(ampl_pcg_data_filtered))

# #for i in range(indices):
# for value in indices_ampl:
#     VTT.append([ (ampl[int(value)])] - (ampl_pcg_data_filtered[2*value]))
#     # print(ampl[int(value) ])
#     # print(int(ampl1[2*value ]))
# indicesVTT = [i for i in range(len(VTT))]
# averageVTT = sum(VTT) / len(VTT)
# plt.figure(" VTT ")
# plt.plot(indicesVTT,VTT)
# plt.xlabel('Số mẫu')
# plt.ylabel('Thời gian VTT')
# plt.title(f'Biểu đồ VTT, Giá trị VTT trung bình: {averageVTT/fs}s' )
# #plt.text(len(VTT), max(VTT), f'Trung bình: {averageVTT}', fontsize=12, ha='center')
# #plt.text(0.1, 0.9, f'Trung bình: {averageVTT}', transform=plt.gcf().transFigure, fontsize=12)
plt.show()
ET = []
# for value in indices_ampl_pcg_data_filtered:
#     if(value % 2 == 0):
#         ET.append(ampl_pcg_data_filtered[value+ 1] - ampl_pcg_data_filtered[value ])
# print(ET)
# ham tim 2 diem gan nhat trong mang
def find_closest_values(arr, target):
    # Tính khoảng cách giữa mỗi giá trị trong mảng và giá trị mục tiêu
    distances = [(abs(value - target), value) for value in arr]

    # Sắp xếp các cặp (khoảng cách, giá trị) theo khoảng cách
    sorted_distances = sorted(distances)

    # Lấy ra hai giá trị có khoảng cách nhỏ nhất
    return sorted_distances[0][1], sorted_distances[1][1]
systolic = []
diastolic = []

for value in ampl:
    sys, dia = find_closest_values(ampl_pcg_data_filtered, value)
    systolic.append(sys)
    diastolic.append(dia)
# print(f'systolic: {systolic}')
# print(f'diastolic: {diastolic}')

# fig, axs = plt.subplots(2, 1, sharex=True)

# axs[0].plot(indices_ppg_data_filtered, median_data)
# axs[0].set_title(" ppg ")
# axs[0].set_xlabel("Số mẫu")
# axs[0].set_ylabel("Giá trị ppg data filter")
# # axs[0].plot(indices, red_movmean_data_flatten)
# for value in ampl:
#      axs[0].plot(value, median_data[value],"r*")

# axs[1].plot(indices_pcg_data_filtered, pcg_filtered)
# axs[1].set_title("pcg")
# axs[1].set_xlabel("Số mẫu")
# axs[1].set_ylabel("Giá trị pcg data fiter")
# axs[1].plot(indices, ir_movmean_data_flatten)
# for value in systolic:
#      axs[1].plot(value, pcg_filtered[value], "y*")
# for value in diastolic:
#      axs[1].plot(value, pcg_filtered[value], "g*")
#         #plt.ylim([30, 120])
VTT = [(s - d) / 500 for d, s in zip(diastolic, systolic)]
ET = [(a - d) / 500 for a, d in zip(ampl, diastolic)]
# fig, axs = plt.subplots(2, 1, sharex=True)

# axs[0].plot(indices_ampl, VTT)
# axs[0].set_title(f'Biểu đồ VTT, Giá trị VTT trung bình: {sum(VTT)/len(VTT)}s')
# axs[0].set_xlabel("Số mẫu")
# axs[0].set_ylabel("Giá trị VTT")
# # axs[0].plot(indices, red_movmean_data_flatten)
# # for value in ampl:
# #      axs[0].plot(value, median_data[value],"r*")

# axs[1].plot(indices_ampl, ET)
# axs[1].set_title(f'Biểu đồ ET, Giá trị ET trung bình: {sum(ET)/len(ET)}s')
# axs[1].set_xlabel("Số mẫu")
# axs[1].set_ylabel("Giá trị ET")
# # axs[1].plot(indices, ir_movmean_data_flatten)
# # for value in systolic:
# #      axs[1].plot(value, pcg_filtered[value], "r*")
# # for value in diastolic:
# #      axs[1].plot(value, pcg_filtered[value], "g*")
# #         #plt.ylim([30, 120])

# plt.tight_layout()
# plt.show()

plt.show()
a1 = 358 
a2 = 13
gamma1 = 0.012164
gamma2 = -1.6281
gamma3 = 71.0961
VTT_a = sum(VTT)/len(VTT)
ET_a = sum(ET)/len(ET)
sbp = a1 * VTT_a + a2
dbp = sbp - gamma1*ET_a/(VTT_a*VTT_a) - gamma2/(VTT_a*VTT_a) - gamma3
print(f'sbp: {sbp}')
print(f'dbp: {dbp}')
print(f'Heart rate: {FHR_average}')

