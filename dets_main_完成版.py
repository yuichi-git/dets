import de_func
import dets_func
import func
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import openpyxl
from tqdm import tqdm

epsilon = 0.5
C = 0.9 #交叉率
F = 0.5 #突然変異係数
D = 2 #次元
N = 5 * D #解候補の数
COUNT = 1000 #試行回数
GENERATION = 100 #世代数
history = 10

# DE+TS用パラメータ
C_ts = [0.9] #交叉率
F_ts = [0.5, 0.9, 1.5] #突然変異係数

num = 4  #手法の数
y_axis = np.zeros((num, GENERATION))
for i in tqdm(range(COUNT)):
    y_axis[0] += de_func.de(C, 0.9, D, GENERATION, func.hump)
    y_axis[1] += de_func.de(C, 0.5, D, GENERATION, func.hump)
    y_axis[2] += dets_func.de_ts(epsilon, C_ts, F_ts, D, GENERATION, func.hump, history)
    y_axis[3] += dets_func.de_ts(0, C_ts, F_ts, D, GENERATION, func.hump, history)

y_axis /= COUNT
x_axis = np.arange(GENERATION)
x_axis += 1

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["font.size"] = 15
plt.rc('legend', fontsize=13)

plt.plot(x_axis, y_axis[0], label="DE F=0.9", linestyle="dashed", color="black")
plt.plot(x_axis, y_axis[1], label="DE F=0.5", linestyle="dashdot", color="black")
plt.plot(x_axis, y_axis[2], label="DE+TS+EP F=[0.5, 0.9, 1.5] TS=10 EP=0.5", linestyle="solid", color="black")
plt.plot(x_axis, y_axis[3], label="DE+TS F=[0.5, 0.9, 1.5] TS=10", linestyle="dotted", color="black")

plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.set_yscale('log')
ax.set_xlabel('Generation', fontname='Times New Roman')
ax.set_ylabel('Value', fontname='Times New Roman')
ax.set_title('hump C=0.9', fontname='Times New Roman')
# plt.savefig("fig/dets/イプシロン無しとdetsの比較/mono_hump_noep_4.pdf")
plt.savefig("test.pdf")
plt.show()