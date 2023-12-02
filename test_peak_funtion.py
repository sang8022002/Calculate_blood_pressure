#ppg_red_data = []
ppg_ir_data = []
fs = 4000
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
#file_name = "pvs_manh_nhe.txt"
# file_name = "pcg_ppg.txt"
file_name = "SNR//400_test.txt"
import csv
windowsize = int(fs/10)
with open(file_name, mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)

    # Duyệt qua từng dòng trong tệp CSV
    for row in csv_reader:
        if len(row) >= 1:  # Đảm bảo có ít nhất 2 cột trong mỗi dòng
            column_0_data = float(row[0])
            # column_1_data = float(row[1])
            #column_6_data = float(row[6])# Lấy dữ liệu từ cột thứ 7 (0-based index)
            # ppg_red_data.append(column_0_data)
            ppg_ir_data.append(column_0_data)
import matplotlib.pyplot as plt
indicies = [i for i in range(len(ppg_ir_data))]
plt.figure("dfrobot data 90g")
plt.plot(indicies, ppg_ir_data)
plt.xlabel("so mau")
plt.ylabel("adc dfrobot")
plt.title("ppg raw from dfrobot")
plt.show()

def movmean_data(A, k):
    x = A.rolling(k,min_periods= 1, center= True).mean().to_numpy()
    return x
def movmedian_data(A, k):
    x = A.rolling(k, min_periods= 1, center= True).median().to_numpy()
    return x
#filter ppg data
ppg_data = ppg_ir_data.copy()
ppg_data_frame = pd.DataFrame(ppg_data)
ppg_data_movmedian = movmedian_data(ppg_data_frame, windowsize)
ppg_data_movmedian = ppg_data_movmedian.flatten()
#fillter pcg data
pcg_data = ppg_ir_data.copy()
import heartpy as hp
pcg_filtered = hp.filter_signal(pcg_data, cutoff = [25, 120], sample_rate = fs,order = 4, filtertype='bandpass')
indicies = [i for i in range(len(pcg_filtered))]
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(indicies, ppg_data)
axs[0].set_xlabel("so mau")
axs[0].set_ylabel("gia tri adc pcg")
axs[0].set_title("PCG raw")

axs[1].plot(indicies, pcg_filtered)
axs[1].set_xlabel("so mau")
axs[1].set_ylabel("gia tri adc pcg")
axs[1].set_title("PCG sau khi loc")

plt.show()


fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(indicies, ppg_ir_data)
axs[0].set_xlabel("so mau")
axs[0].set_ylabel("gia tri adc ppg")
axs[0].set_title("PPG raw")

axs[1].plot(indicies, ppg_data_movmedian)
axs[1].set_xlabel("so mau")
axs[1].set_ylabel("gia tri adc ppg")
axs[1].set_title("PPG sau khi loc")

plt.show()




