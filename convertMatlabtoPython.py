import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

# Clear allq
plt.close('all')
plt.clf()
plt.cla()

# Load CSV file
file_name = "IR_RED_LED.txt"
ch = 1
fs = 25
colum_red = "red"
colum_ir = "ir"
A3 = pd.read_csv(file_name,usecols= [colum_ir]).to_numpy()
ArrayRed = pd.read_csv(file_name,usecols= [colum_red]).to_numpy()

# A3 = pd.read_csv(file_name)
A1 = A3.copy()
A2 = pd.DataFrame(A1)
ArrayRed1 = ArrayRed.copy()
ArrayRed2 = pd.DataFrame(ArrayRed1)


# Remove NaN and Zero values
x = np.arange(1, len(A1[:, 0]) + 1)
x1 = np.arange(1, len(ArrayRed1[:, 0]) + 1)
plt.figure(90)

for i in range(1, ch + 1):
    plt.subplot(ch, 1, i)
    plt.plot(x / fs, A1[:, i - 1])
plt.show()

nan_idx = np.isnan(A1)
A1[nan_idx] = 1e-16

zero_idx = (A1 == 0)
A1[zero_idx] = 1e-16
#calculate red
nan_idx = np.isnan(ArrayRed1)
ArrayRed1[nan_idx] = 1e-16

zero_idx = (ArrayRed1 == 0)
ArrayRed1[zero_idx] = 1e-16
#################################

def movmean1(A, k):
    x = A.rolling(k,min_periods= 1, center= True).mean().to_numpy()
    return x
def movmedian1(A, k):
    x = A.rolling(k, min_periods= 1, center= True).median().to_numpy()
    return x
#calculate ac/dc red
baseline_data_red = movmean1(ArrayRed2, fs)
acDivDcRed = ArrayRed1/baseline_data_red

##############################################

baseline_data = movmean1(A2, fs)
#clean_data = A1[:, 0] - baseline_data
# acDivDcIr = A1/baseline_data
# plt.figure(20)
# for i in range(1, ch + 2):
#     if i != (ch + 1):
#         plt.subplot(ch+1, 1, i)
#         plt.plot(x/fs, acDivDcIr)
#         # plt.plot(x/fs, )
#     else:
#         plt.subplot(ch+1, 1, i)
#         plt.plot(x/fs, acDivDcRed)
# plt.show()
# R = acDivDcRed/acDivDcIr
# spo2 = 110-25*R
#
# plt.figure("spo2")
# plt.plot(x/fs, spo2)
# plt.show()

clean_data = (A1 - baseline_data)
window_size = int(fs / 10)
clean_data = pd.DataFrame(clean_data )
median_data = movmedian1(clean_data,window_size )

clean_data_invert = baseline_data - A1
# window_size = int(fs / 10)
clean_data_invert = pd.DataFrame(clean_data_invert)
median_data_invert = movmedian1(clean_data_invert,window_size )

clean_data_red = (ArrayRed1 - baseline_data_red)
window_size = int(fs / 10)
clean_data_red = pd.DataFrame(clean_data_red )
median_data_red = movmedian1(clean_data_red,window_size )

clean_data_red_invert = (- ArrayRed1 + baseline_data_red)
#window_size = int(fs / 10)
clean_data_red_invert = pd.DataFrame(clean_data_red_invert )
median_data_red_invert= movmedian1(clean_data_red_invert,window_size )
plt.figure(9000)

for i in range(1, ch + 1):
    plt.subplot(ch, 1, i)
    plt.plot(x / fs, A1[:, i - 1])
    plt.plot(x / fs, baseline_data)
plt.show() #code heare

plt.figure(9001)


for i in range(1, ch + 1):
    plt.subplot(ch, 1, i)
    plt.plot(x / fs, median_data)
plt.show()

# Find peaks

df = pd.DataFrame(median_data)

# Trích xuất cột 'median_data' thành mảng NumPy 1 chiều
##tinh cho ir
median_data4 = median_data.flatten()
median_data5 = median_data_invert.flatten()
ampl, __ = find_peaks(median_data4, distance=int(0.7 * fs))
ampl_invert, __ = find_peaks(median_data5, distance=int(0.7 * fs))
##tinh cho red
median_data_red_4 = median_data_red.flatten()
median_data_red_5 = median_data_red_invert.flatten()
ampl_red, __ = find_peaks(median_data_red_4, distance=int(0.7 * fs))
ampl_red_invert, __ = find_peaks(median_data_red_5, distance=int(0.7 * fs))

######## Tính giá trị giữa dinh - day #############################################################
###################################################################################################
plt.figure("find peak - peak ir")

for i in range(1, ch + 2):
    if i != (ch + 1):
        plt.subplot(ch + 1, 1, i)
        plt.plot(x, median_data)
        for value in ampl:
            plt.plot(value, median_data[value], "x")
    else:
        plt.subplot(ch + 1, 1, i)
        plt.plot(x, median_data_invert)
        for value in ampl_invert:
            plt.plot(value, median_data_invert[value], "x")
plt.show()
ac_amplitute_ir = median_data4[ampl] + median_data5[ampl_invert]
plt.figure("ac ampitute")
print(ac_amplitute_ir)
plt.plot(ampl, ac_amplitute_ir)
plt.show()
#tim diem nhỏ nhat giữa các dinhampl và ampl_invert
min_points = []

for i in range(len(ampl)):
    peak_start = ampl[i]  # Vị trí bắt đầu của đỉnh hiện tại
    if i < len(ampl_invert):
        peak_end = ampl_invert[i]  # Vị trí kết thúc của đỉnh đối diện
        # Tìm giá trị nhỏ nhất trong khoảng giữa hai đỉnh
        min_value = np.min(baseline_data[peak_start:peak_end+1])
        min_points.append(min_value)
print(min_points)
ac_div_dc_ir = ac_amplitute_ir/min_points

min_points1 = []
ac_amplitute_red = median_data_red_4[ampl_red] + median_data_red_5[ampl_red_invert]



for i in range(len(ampl_red)):
    peak_start = ampl_red[i]  # Vị trí bắt đầu của đỉnh hiện tại
    if i < len(ampl_red_invert):
        peak_end = ampl_red_invert[i]  # Vị trí kết thúc của đỉnh đối diện
        # Tìm giá trị nhỏ nhất trong khoảng giữa hai đỉnh
        min_value = np.min(baseline_data_red[peak_start:peak_end+1])
        min_points1.append(min_value)
ac_div_dc_red = ac_amplitute_red/min_points1

ror = ac_div_dc_ir/ac_div_dc_red
indices_ror = [i for i in range(len(ror))]
spo2 = 110 - 25*ror

#######################################################################
fig, axs = plt.subplots(2, 2, sharex=True)
axs[0,0].plot(indices_ror, ac_amplitute_ir)
axs[0,0].set_title("ac ir")

axs[0,1].plot(indices_ror, ac_amplitute_red)
axs[0,1].set_title("ac red")
# axs[1].plot(indices, ir_movmean_data_flatten)
axs[1,0].plot(indices_ror, min_points)
axs[1,0].set_title("dc ir")

axs[1,1].plot(indices_ror, min_points1)
axs[1,1].set_title("dc red")

fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(indices_ror, ac_amplitute_ir/min_points)
axs[0].set_title("ac ir div dc")

axs[1].plot(indices_ror, ac_amplitute_red/min_points1)
axs[1].set_title("ac red div dc")
########################################################################################
fig, axs = plt.subplots(1, 1, sharex=True)
axs.plot(indices_ror, ror)
axs.set_title("ror")
plt.show()

########################################################################################
########################################################################################

duong_bao = []

plt.figure(9002)
for i in range(1, ch + 1):
    plt.subplot(ch, 1, i)
    plt.plot(x, median_data)
    for value in ampl:
        plt.plot(value, median_data[value], "r*")
        duong_bao.append([value, median_data[value]])
# Vẽ đường bao
duong_bao_x = [point[0] for point in duong_bao]
duong_bao_y = [point[1] for point in duong_bao]
plt.plot(duong_bao_x, duong_bao_y, 'b-', label='Đường bao')

# Thêm tiêu đề và nhãn cho đồ thị
plt.title('Đường bao của các điểm đỏ')
plt.xlabel('Giá trị x')
plt.ylabel('Giá trị trung vị')
plt.grid(True)
plt.legend()
plt.show()



# Calculate FHR = 60*fs/RR
RR = ampl[1:] - ampl[:-1]
FHR = 60 * fs / RR
print(FHR)
plt.figure(9003)

for i in range(1, ch + 2):
    if i != (ch + 1):
        plt.subplot(ch + 1, 1, i)
        plt.plot(x, median_data)
        for value in ampl:
            plt.plot(value, median_data[value], "x")
    else:
        plt.subplot(ch + 1, 1, i)
        plt.plot(ampl[:-1], FHR)
        plt.ylim([30, 120])
plt.show()
# clean_data0 = clean_data
median_data = pd.DataFrame(median_data )
baseline_data1 = movmean1(median_data, fs)
median_data = np.array(median_data)
baseline_data1 = np.array(baseline_data1)
median_data1 = - median_data + baseline_data1
median_data1 = median_data1.flatten()
plt.figure(9004)
for i in range(1, ch + 2):
    if i != (ch + 1):
        plt.subplot(ch+1, 1, i)
        plt.plot(x/fs, median_data)
        plt.plot(x/fs, baseline_data1)
    else:
        plt.subplot(ch+1, 1, i)
        plt.plot(x/fs, median_data1)
plt.show()
ampl1, __ = find_peaks(median_data1, distance=int(0.7 * fs))

plt.figure(9005)

for i in range(1, ch + 1):
    plt.subplot(ch, 1, i)
    plt.plot(x, median_data1)
    for value in ampl1:
        plt.plot(value, median_data1[value], "r*")
plt.show()





