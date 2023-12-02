ppg_red_data = []
ppg_ir_data = []
fs = 100
from scipy.signal import find_peaks
#file_name = "pvs_manh_nhe.txt"
# file_name = "pcg_ppg.txt"
file_name = "ngontay-Ha-1700145820-100.csv"
import csv
windowsize = int(fs/10)
with open(file_name, mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)

    # Duyệt qua từng dòng trong tệp CSV
    for row in csv_reader:
        if len(row) >= 2:  # Đảm bảo có ít nhất 2 cột trong mỗi dòng
            column_0_data = float(row[0])
            column_1_data = float(row[1])
            #column_6_data = float(row[6])# Lấy dữ liệu từ cột thứ 7 (0-based index)
            ppg_red_data.append(column_0_data)
            ppg_ir_data.append(column_1_data)
            # Sử dụng dữ liệu từ cột thứ 7 ở đây, ví dụ:
            #print(column_data)
indices = [i for i in range(len(ppg_red_data))]
import matplotlib.pyplot as plt
plt.figure("ppg ir")
plt.title("PPG IR RAW")
plt.xlabel("Sampling")
plt.ylabel("ADC value")
plt.plot(indices,ppg_ir_data)
plt.show()
ch = 1
plt.figure("ppg red")
plt.tight_layout()
for i in range(1, ch + 2):
    if i != (ch + 1):
        plt.subplot(ch + 1, 1, i)
        plt.xlabel("So mẫu")
        plt.ylabel("Gia trị ADC của red")
        plt.title("ppg red data")
        plt.plot(indices, ppg_red_data)
    else:
        plt.subplot(ch + 1, 1, i)
        plt.xlabel("So mẫu")
        plt.ylabel("Gia trị ADC của ir")
        plt.title("ppg ir data")
        plt.plot(indices, ppg_ir_data)
plt.tight_layout()
plt.show()
def movmean_data(A, k):
    x = A.rolling(k,min_periods= 1, center= True).mean().to_numpy()
    return x
def movmedian_data(A, k):
    x = A.rolling(k, min_periods= 1, center= True).median().to_numpy()
    return x
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

ppg_ir_data_copy = ppg_ir_data
ppg_red_data_copy = ppg_red_data

ppg_ir_data_frame = pd.DataFrame(ppg_ir_data_copy)
ppg_red_data_frame = pd.DataFrame(ppg_red_data_copy)

windowsize = int(0.1*fs)

ir_median_data = movmedian_data(ppg_ir_data_frame, windowsize)
red_median_data = movmedian_data(ppg_red_data_frame, windowsize)

ir_median_data_frame = pd.DataFrame(ir_median_data)
red_median_data_frame = pd.DataFrame(red_median_data)

ir_movmean_data_frame = movmean_data(ir_median_data_frame, fs)
red_movmean_data_frame = movmean_data(red_median_data_frame, fs)

ir_movmean_data_flatten = ir_movmean_data_frame.flatten()
red_movmean_data_flatten = red_movmean_data_frame.flatten()
print(f'red_movmean_data_flatten: {red_movmean_data_flatten}')
# ir_median_data_flatten = ir_median_data_frame.flatten()
# red_median_data_flatten = red_median_data_frame.flatten()

ir_median_data = ir_median_data.flatten()
red_median_data = red_median_data.flatten()

indices = [i for i in range(len(ppg_ir_data_copy))]

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].plot(indices, red_median_data_frame)
axs[0].plot(indices, red_movmean_data_flatten)
axs[0].set_title("ppg red signal after filtering ")
axs[0].set_xlabel("sampling")
axs[0].set_ylabel("ADC value")

axs[1].set_title("ppg ir signal after filtering")
axs[1].set_xlabel("sampling")
axs[1].set_ylabel("ADC value")
axs[1].plot(indices, ir_median_data_frame)
axs[1].plot(indices, ir_movmean_data_flatten)

plt.tight_layout()
plt.show()

# ################################################################################
ampl_red, __= find_peaks(red_median_data, distance=int(0.5 * fs) )
ampl_ir, __= find_peaks(ir_median_data, distance=int(0.5 * fs) )

# ####################################################################################
# test rolling with ampl
def caculate_heart_rate(data):
    ampl_, __= find_peaks(data, distance=int(0.5 * fs) )
    
    RR = ampl_[1:] - ampl_[:-1]
    FHR = 60 * fs / RR  # ví dụ, gấp đôi tổng của cửa sổ
    FHR_average = np.mean(FHR)
    return FHR_average

# Tạo một DataFrame
ampl_red_frame = pd.DataFrame(red_median_data)
ampl_ir_frame = pd.DataFrame(ir_median_data)
# Áp dụng hàm rolling với cửa sổ trượt có kích thước 3 và áp dụng hàm custom_function
heart_rate_red = ampl_red_frame.rolling(window = 400, min_periods= 1, center= True).apply(caculate_heart_rate, raw=True)
# print(heart_rate_red)
heart_rate_ir = ampl_ir_frame.rolling(window = 400, min_periods= 1, center= True).apply(caculate_heart_rate, raw=True)
# ################################################################################
# heart_rate = heart_rate.flatten()

indices_heart_rate = [i for i in range(len(heart_rate_red))]

fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(indices_heart_rate, heart_rate_red)
axs[0].set_title("rolling heart rate")
axs[0].set_xlabel("sampling")
axs[0].set_ylabel("heart rolling with red data")

axs[1].plot(indices_heart_rate, heart_rate_ir)
axs[1].set_title("rolling heart rate ")
axs[1].set_xlabel("sampling")
axs[1].set_ylabel("heart rolling with ir data")
plt.tight_layout()

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].plot(indices, red_median_data_frame)
axs[0].set_title("RED")
axs[0].set_xlabel("Số mẫu")
axs[0].set_ylabel("Giá trị ADC")
# axs[0].plot(indices, red_movmean_data_flatten)
for value in ampl_red:
     axs[0].plot(value, red_median_data[value],"r*")
axs[1].plot(indices, ir_median_data)
axs[1].set_title("IR")
axs[1].set_xlabel("Số mẫu")
axs[1].set_ylabel("Giá trị ADC")
for value in ampl_ir:
     axs[1].plot(value, ir_median_data[value], "r*")
        #plt.ylim([30, 120])
plt.tight_layout()
plt.show()

ac_red = (ppg_red_data_copy - red_movmean_data_flatten)
ac_red_invert = -(ppg_red_data_copy - red_movmean_data_flatten)

ac_ir = (ppg_ir_data_copy - ir_movmean_data_flatten)
ac_ir_invert = -(ppg_ir_data_copy - ir_movmean_data_flatten)

ac_red_frame = pd.DataFrame(ac_red)
ac_red_invert_frame = pd.DataFrame(ac_red_invert)
ac_ir_frame = pd.DataFrame(ac_ir)
ac_ir_invert_frame = pd.DataFrame(ac_ir_invert)

ac_red_median_data = movmedian_data(ac_red_frame, windowsize)
ac_red_median_invert_data = movmedian_data(ac_red_invert_frame, windowsize)
ac_ir_median_data = movmedian_data(ac_ir_frame, windowsize)
ac_ir_median_invert_data = movmedian_data(ac_ir_invert_frame, windowsize)

ac_ir_median_data = ac_ir_median_data.flatten()
ac_ir_median_invert_data = ac_ir_median_invert_data.flatten()
ac_red_median_data = ac_red_median_data.flatten()
ac_red_median_invert_data = ac_red_median_invert_data.flatten()

ampl_ac_red, __= find_peaks(ac_red_median_data, distance=int(0.5 * fs),width= 0.08*fs )
ampl_ac_red_invert,__ = find_peaks(ac_red_median_invert_data, distance=int(0.5 * fs), width= 0.08*fs )
ampl_ac_ir, __= find_peaks(ac_ir_median_data, distance=int(0.5 * fs), width= 0.08*fs )
ampl_ac_ir_invert,__ = find_peaks(ac_ir_median_invert_data, distance=int(0.5 * fs), width= 0.01*fs )



fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].plot(indices, ac_red_median_data)
axs[0].set_title(" AC RED ")
axs[0].set_xlabel("Số mẫu")
axs[0].set_ylabel("Giá trị ADC RED")
# axs[0].plot(indices, red_movmean_data_flatten)
for value in ampl_ac_red:
     axs[0].plot(value, ac_red_median_data[value],"r*")

axs[1].plot(indices, ac_ir_median_data)
axs[1].set_title("AC IR")
axs[1].set_xlabel("Số mẫu")
axs[1].set_ylabel("Giá trị ADC IR")
# axs[1].plot(indices, ir_movmean_data_flatten)
for value in ampl_ac_ir:
     axs[1].plot(value, ac_ir_median_data[value], "r*")
        #plt.ylim([30, 120])
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 2, sharex=True)

axs[0,0].plot(indices, ac_red_median_data)
axs[0,0].set_title("RED")
# axs[0].plot(indices, red_movmean_data_flatten)
for value in ampl_ac_red:
     axs[0,0].plot(value, ac_red_median_data[value],"r*")
axs[1,0].plot(indices, ac_red_median_invert_data)
axs[1,0].set_title("RED invert")
# axs[0].plot(indices, red_movmean_data_flatten)
for value in ampl_ac_red_invert:
     axs[1,0].plot(value, ac_red_median_invert_data[value],"r*")
        # for value in ampl:
        # plt.plot(value, median_data[value], "x")
    # else:
        # plt.subplot(ch + 1, 1, i)
        # plt.plot(indices, pcg_data)
axs[0,1].plot(indices, ac_ir_median_data)
axs[0,1].set_title("IR")
# axs[0].plot(indices, red_movmean_data_flatten)
for value in ampl_ac_ir:
     axs[0,1].plot(value, ac_ir_median_data[value],"r*")
axs[1,1].plot(indices, ac_ir_median_invert_data)
axs[1,1].set_title("IR invert")
# axs[1].plot(indices, ir_movmean_data_flatten)
for value in ampl_ac_ir_invert:
     axs[1,1].plot(value, ac_ir_median_invert_data[value], "r*")
        #plt.ylim([30, 120])
plt.tight_layout()
plt.show()

ampl_ac_red_copy = ampl_ac_red
ampl_ac_red_invert_copy = ampl_ac_red_invert
ampl_ac_ir_copy = ampl_ac_ir
ampl_ac_ir_invert_copy = ampl_ac_ir_invert

if ampl_ac_red_copy[0] > ampl_ac_red_invert_copy[0]:
    ampl_ac_red_invert_copy = ampl_ac_red_invert_copy[1:]
if ampl_ac_red_copy[-1] > ampl_ac_red_invert_copy[-1]:
    print(ampl_ac_red_copy)
    ampl_ac_red_copy = ampl_ac_red_copy[:(len(ampl_ac_red_copy) - 1)]
    print(ampl_ac_red_copy)
if ampl_ac_ir_copy[0] > ampl_ac_ir_invert_copy[0]:
    ampl_ac_ir_invert_copy = ampl_ac_ir_invert_copy[1:]
if ampl_ac_ir_copy[-1] > ampl_ac_ir_invert_copy[-1]:
    ampl_ac_ir_copy = ampl_ac_ir_copy[:(len(ampl_ac_ir_copy) - 1)]
fig, axs = plt.subplots(2, 2, sharex=True)

axs[0,0].plot(indices, ac_red_median_data)
axs[0,0].set_title("RED1")
# axs[0].plot(indices, red_movmean_data_flatten)
for value in ampl_ac_red_copy:
     axs[0,0].plot(value, ac_red_median_data[value],"r*")
axs[1,0].plot(indices, ac_red_median_invert_data)
axs[1,0].set_title("RED invert")
# axs[0].plot(indices, red_movmean_data_flatten)
for value in ampl_ac_red_invert_copy:
     axs[1,0].plot(value, ac_red_median_invert_data[value],"r*")
        # for value in ampl:
        # plt.plot(value, median_data[value], "x")
    # else:
        # plt.subplot(ch + 1, 1, i)
        # plt.plot(indices, pcg_data)
axs[0,1].plot(indices, ac_ir_median_data)
axs[0,1].set_title("IR")
# axs[0].plot(indices, red_movmean_data_flatten)
for value in ampl_ac_ir_copy:
     axs[0,1].plot(value, ac_ir_median_data[value],"r*")
axs[1,1].plot(indices, ac_ir_median_invert_data)
axs[1,1].set_title("IR invert")
# axs[1].plot(indices, ir_movmean_data_flatten)
for value in ampl_ac_ir_invert_copy:
     axs[1,1].plot(value, ac_ir_median_invert_data[value], "r*")
        #plt.ylim([30, 120])
plt.tight_layout()
plt.show()


ac_strip_red = ac_red_median_data[ampl_ac_red_copy] + ac_red_median_invert_data[ampl_ac_red_invert_copy]
ac_strip_ir = ac_ir_median_data[ampl_ac_ir_copy] + ac_ir_median_invert_data[ampl_ac_ir_invert_copy]


indices_ampl_ac_red = [i for i in range(len(ac_strip_red))]
indices_ampl_ac_ir = [i for i in range(len(ac_strip_ir))]



min_dc_red =[]
for i in range(len(ampl_ac_red)):
    # peak_start = ampl_ac_red[i]  # Vị trí bắt đầu của đỉnh hiện tại
    peak_start = ampl_red[i]
    if i < len(ampl_ac_red_invert):
        #peak_end = ampl_red_invert[i]  # Vị trí kết thúc của đỉnh đối diện
        # Tìm giá trị nhỏ nhất trong khoảng giữa hai đỉnh
        peak_end = ampl_ac_red_invert[i]
        min_value = np.min(red_movmean_data_flatten[peak_start:peak_end + 1])
        min_dc_red.append(min_value)   

min_dc_ir =[]
for i in range(len(ampl_ac_ir)):
    # peak_start = ampl_ac_ir[i]  # Vị trí bắt đầu của đỉnh hiện tại
    peak_start = ampl_red[i]
    if i < len(ampl_ac_ir_invert):
        peak_end = ampl_ac_ir_invert[i]
        #peak_end = ampl_red_invert[i]  # Vị trí kết thúc của đỉnh đối diện
        # Tìm giá trị nhỏ nhất trong khoảng giữa hai đỉnh
        min_value = np.min(ir_movmean_data_flatten[peak_start:peak_end + 1])
        min_dc_ir.append(min_value)   

ac_div_dc_red = ac_strip_red / min_dc_red
ac_div_dc_ir = ac_strip_ir / min_dc_ir
ror = ac_div_dc_ir/ ac_div_dc_red
indices_spo2 = [i for i in range(len(ror))]
spo2 = 110 - 25*ror

fig, axs = plt.subplots(2, 2, sharex=True)
axs[0,0].plot(indices_spo2, ac_strip_ir)
axs[0,0].set_title("ac ir")

axs[0,1].plot(indices_spo2, ac_strip_red)
axs[0,1].set_title("ac red")
# axs[1].plot(indices, ir_movmean_data_flatten)
axs[1,0].plot(indices_spo2, min_dc_ir)
axs[1,0].set_title("dc ir")

axs[1,1].plot(indices_spo2, min_dc_red)
axs[1,1].set_title("dc red")

fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(indices_spo2, ac_strip_ir/min_dc_ir)
axs[0].set_title("ac ir div dc")

axs[1].plot(indices_spo2, ac_strip_red/min_dc_red)
axs[1].set_title("ac red div dc")

fig, axs = plt.subplots(1, 1, sharex=True)
axs.plot(indices_spo2, ror)
average_ror = sum(ror)/len(ror)
axs.set_title(f'Biểu đồ, Giá trị ror trung bình: {average_ror}s')
# plt.title(f'Biểu đồ VTT, Giá trị VTT trung bình: {averageVTT/fs}s' )
plt.tight_layout()
plt.show()

plt.figure("spo2")
plt.plot(indices_spo2, spo2)
plt.xlabel("So mau")
plt.ylabel(" '%' Spo2")
plt.title(f'Spo2, Average spo2 = {sum(spo2)/len(spo2)}')
plt.grid
plt.show()


