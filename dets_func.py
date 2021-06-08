import de_func
import random
from copy import deepcopy
import copy
import pandas as pd
import numpy as np
from scipy.special import gamma
import csv
import pprint

def ts(n, hit, miss):
    sample = np.zeros(n)
    #各アームのベータ分布に基づいて乱数を出す
    for k in range(n):
        sample[k] = np.random.beta(hit[k]+1, miss[k]+1, 1)
    #乱数が一番大きいアームを選択
    selectArm = np.argmax(sample)
    return selectArm

def de_ts_gene(epsilon, C, F, x_list, func, c_hm, f_hm, c_hm_num, f_hm_num, g, history):
    X_MAX = func('max')
    X_MIN = func('min')
    N = x_list.shape[0]
    D = x_list.shape[1]

    sel_f_0 = 0
    sel_f_1 = 0
    sel_f_2 = 0

    #step2-1 ランダムに選んだ３つのベクトルを用いて新たなベクトルvを計算、という作業を解候補ごとに行う
    for i in range(N):

        c_hm_num = np.sum(c_hm, axis=2)
        f_hm_num = np.sum(f_hm, axis=2)
        # print("sum")
        # print(f_hm_num)
        # print(f_hm_num[:, 0])
        # print(f_hm_num[:, 1])

        # イプシロングリーディー
        rand_n = np.random.rand()
        if rand_n >= epsilon:
            # トンプソンサンプリングで使うパラメータを決定
            select_c = ts(len(C), c_hm_num[:, 0], c_hm_num[:, 1])
            select_f = ts(len(F), f_hm_num[:, 0], f_hm_num[:, 1])
        else:
            select_c = np.random.randint(0, len(C))
            select_f = np.random.randint(0, len(F))

        if select_f == 0:
            sel_f_0 += 1
        elif select_f == 1:
            sel_f_1 += 1
        elif select_f == 2:
            sel_f_2 += 1

        # print("hit ", f_hm_num[:, 0])
        # print("miss", f_hm_num[:, 1], "next F", select_f)
        # print() #当たり外れの履歴チェック
        number = np.arange(N) #解候補の何番目をとるか決めるための、0~N-1の数が入った配列を作成
        number = np.delete(number, i)           #ターゲットベクトルの番号は削除
        selected_number = np.random.choice(number, 3, replace = False) #ターゲットベクトル以外から重複なしで３つの解候補を抽出

        base_x = deepcopy(x_list[selected_number[0]]) #抽出した１つをベースベクトルとする
        x_a = deepcopy(x_list[selected_number[1]])
        x_b = deepcopy(x_list[selected_number[2]])
        v = base_x + F[select_f] * (x_b - x_a) #vの計算

        #step2-2 vを用いて新たなベクトルzを作る
        z = np.zeros(D) #解候補になるかもしれないベクトル
        # jr = np.random.randint(0, D) #0~D-1の添字をランダムに選ぶ
        for j in range(D):
            n_1 = np.random.rand() #0~1の実数をランダムで生成
            if n_1 <= C[select_c]: #条件に当てはまれば、zの要素にvの要素を代入
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
            c_hm[select_c][0][g*N+i] = 1
            c_hm[select_c][1][g*N+i] = 0
            f_hm[select_f][0][g*N+i] = 1
            f_hm[select_f][1][g*N+i] = 0
        else:
            c_hm[select_c][0][g*N+i] = 0
            c_hm[select_c][1][g*N+i] = 1
            f_hm[select_f][0][g*N+i] = 0
            f_hm[select_f][1][g*N+i] = 1
        if g*N+i >= history:
            for j in range(len(C)):
                for k in range(2):
                    c_hm[j][k][g*N+i-history] = 0
            for j in range(len(F)):
                for k in range(2):
                    f_hm[j][k][g*N+i-history] = 0
        # f_hm *= 0.5
        # print(select_f)
        # print(f_hm)
        # print()

def de_ts(epsilon, C, F, D, GENERATION, func, history):
    N = 5 * D
    min_solution = np.zeros(GENERATION)
    X_MAX = func('max')
    X_MIN = func('min')
    x_list = np.random.rand(N, D) * (X_MAX - X_MIN) + X_MIN
    c_hm = np.zeros((len(C), 2, GENERATION*N))
    f_hm = np.zeros((len(F), 2, GENERATION*N))
    c_hm_num = np.zeros((len(C), 2))
    f_hm_num = np.zeros((len(F), 2))
    f_ts_result = [[] for i in range(len(F))]

    for i in range(GENERATION):
        de_ts_gene(epsilon, C, F, x_list, func, c_hm, f_hm, c_hm_num, f_hm_num, i, history)
        min_solution[i] = de_func.min_sol(x_list, func)
        tmp = deepcopy(np.sum(f_hm, axis=2))
        for j in range(len(F)):
            f_ts_result[j].append(list(tmp[j]))

    return min_solution