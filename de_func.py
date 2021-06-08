import numpy as np
from copy import deepcopy
import copy
from tqdm import tqdm
import math
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# 解候補から最小解を求める
def min_sol(x_list, func):
    N = x_list.shape[0]
    sol = np.zeros(N)
    for i in range(N):
        sol[i] = deepcopy(func(x_list[i]))
    return np.min(sol)

def de_gene(C, F, x_list, func):
    X_MAX = func('max')
    X_MIN = func('min')
    N = x_list.shape[0]
    D = x_list.shape[1]
    #step2-1 ランダムに選んだ３つのベクトルを用いて新たなベクトルvを計算、という作業を解候補ごとに行う
    for i in range(N):
        number = np.arange(N) #解候補の何番目をとるか決めるための、0~N-1の数が入った配列を作成
        number = np.delete(number, i)           #ターゲットベクトルの番号は削除
        selected_number = np.random.choice(number, 3, replace = False) #ターゲットベクトル以外から重複なしで３つの解候補を抽出

        base_x = deepcopy(x_list[selected_number[0]]) #抽出した１つをベースベクトルとする
        x_a = deepcopy(x_list[selected_number[1]])
        x_b = deepcopy(x_list[selected_number[2]])
        v = base_x + F * (x_b - x_a) #vの計算

        #step2-2 vを用いて新たなベクトルzを作る
        z = np.zeros(D) #解候補になるかもしれないベクトル
        # jr = np.random.randint(D) #0~D-1の添字をランダムに選ぶ
        for j in range(D):
            n_1 = np.random.rand() #0~1の実数をランダムで生成
            if n_1 <= C: #条件に当てはまれば、zの要素にvの要素を代入
                if X_MIN <= v[j] and v[j] <= X_MAX:
                    z[j] = copy.deepcopy(v[j])
                elif v[j] <= X_MIN: #定義域から外れた時の処理
                    z[j] = base_x[j] + np.random.rand() * (X_MIN - base_x[j])
                elif v[j] >= X_MAX:
                    z[j] = base_x[j] + np.random.rand() * (X_MAX - base_x[j])
            else:
                z[j] = copy.deepcopy(x_list[i][j]) #条件を満たさなければ、xの要素をそのまま代入
        #step2-3 zの方が良い解ならば、zを解候補に加える
        if func(x_list[i]) > func(z):
            x_list[i] = deepcopy(z)

def de(C, F, D, GENERATION, func):
    N = 5 * D
    min_solution = np.zeros(GENERATION)
    X_MAX = func('max')
    X_MIN = func('min')
    #step1 初期値の設定
    x_list = np.random.rand(N, D) * (X_MAX - X_MIN) + X_MIN
    for i in range(GENERATION):
    #step2-1 ランダムに選んだ３つのベクトルを用いて新たなベクトルvを計算、という作業を解候補ごとに行う
        de_gene(C, F, x_list, func)
    #step3 一番良い解を求める
        min_solution[i] = min_sol(x_list, func)
    return min_solution

def de_scatter(frame, C, F, x_list, func):
    plt.cla()
    de_gene(C, F, x_list, func)
    X_MAX = func('max')
    X_MIN = func('min')
    x, y = zip(*x_list)
    plt.scatter(x, y)
    plt.xlim(X_MIN, X_MAX)
    plt.ylim(X_MIN, X_MAX)
    # gif_range = 0.001
    # plt.xlim(-gif_range, gif_range)
    # plt.ylim(-gif_range, gif_range)
    plt.grid(True)

def de_gif(C, F, D, func):
    X_MAX = func('max')
    X_MIN = func('min')
    N = 5 * D
    x_list = np.random.rand(N, D) * (X_MAX - X_MIN) + X_MIN
    fig = plt.figure()
    ani = animation.FuncAnimation(fig, de_scatter, fargs=(C, F, x_list, func), interval=100, frames=15)
    ani.save("fig/de_anime.gif", writer="imagemagick")
    plt.show()