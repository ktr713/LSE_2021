#2019/11/19~ Written by ryota kato
import numpy as np
import json
from numba import jit,i8,f8

#変数等をまとめる

#-------------------------------------------------------定数--------------------------------------------------------------------#
myu = 3.986*10**5
myu_L = 4.90278*10**3
M_Moon = 7.348020*10**22
moon_radius = 1737.0
r_L = 384748.0
#omega = (myu/r_L**3)**(1/2)
omega = 2.6616*10**-6
cr = 0.1
theta = 6.68    #軌道傾斜角
ecc = 0.0527    #離心率
alpha = 393466.0174
vmax = 0.083333
vmin = 0.2
accel = 3.85*10**-8
Mass_climber = 10000.0
v_tether_max = 0.03
v_tether_min = 0.00001
k_max_rate = 5.0

t_return = 27.3217/2*3600*24*10

#------------------------------------------------------入力処理-----------------------------------------------------------------#

#f1 = open("inputdata\\CNT\\datas.json","r")
#f2 = open("inputdata\\CNT\\pars.json","r")
#tapers = open("inputdata\\CNT\\Areas.csv","r")

f1 = open("inputdata\\zylon\\perigee\\datas.json","r")
f2 = open("inputdata\\zylon\\perigee\\pars.json","r")
tapers = open("inputdata\\zylon\\perigee\\Areas.csv","r")

datas = json.load(f1)
pars = json.load(f2)

E = float(datas["young_ratio"])
sigma = float(datas["sigma"])
rho = float(datas["rho"])
N = int(datas["N"])
L = float(datas["tether_Length"])
Mcw = float(pars["Mcw"])
t_L = float(pars["el_Length"])
material = datas["Name"]

Areas = tapers.readlines()
Areas = [float(n) for n in Areas]
Areas = np.array(Areas)

f1.close()
f2.close()
tapers.close()

#計算値
sigma_par_p = sigma*0.95/(10**3)
sigma_par_m = sigma*0.9/(10**3)
l_0 = t_L/float(N)
l_0_short = l_0/10.0
k_ratios = E/(l_0*10**3) * Areas    
masses = rho * Areas * l_0*10**3
damps = 2*(masses*k_ratios)**(1/2)*cr

l_0_min = l_0/1000.0

#------------------------------------------------------操作関数-----------------------------------------------------------------#

#実行時間 日->秒
def timer(s,dt):
    return int(s*3600*24*1/dt)


#分割用関数(初期状態の設計)
def divide_vectors(vectors,case=0):
    #月面操作
    if(case == 0):
        y1 = vectors[0,1]
        y2 = vectors[1,1]
        y3 = vectors[2,1]

        #追加行列
        addvector = np.zeros((9,6))
        #0~1間を10等分:1->10なので1~9の9質点を用意
        for i in range(9):
            addvector[i,1] = (y2-y1)/10*(i+1) + y1

        vectors = np.insert(vectors,1,addvector,axis=0)

        #1~2間を10等分:1->10,2->20なので11~19の9質点を用意
        for i in range(9):
            addvector[i,1] = (y3-y2)/10*(i+1) + y2
            addvector[i,4] = 2

        vectors = np.insert(vectors,11,addvector,axis=0)

    #CW操作
    elif(case == 1):
        y1 = vectors[-2,1]
        y2 = vectors[-3,1]
        y3 = vectors[-4,1]

        #追加行列
        addvector = np.zeros((9,6))
        #CW~CW-1間を10等分:-3->-11なので-2~-10の9質点を用意
        for i in range(9):
            addvector[i,1] = (y1-y2)/10*(i+1) + y2
            addvector[i,4] = 2
        vectors = np.insert(vectors,-2,addvector,axis=0)

        #CW-1~CW-2間を10等分:-3->-11,-4->-21なので-12~-20の9質点を用意
        for i in range(9):
            addvector[i,1] = (y2-y3)/10*(i+1) + y3
            addvector[i,4] = 2

        vectors = np.insert(vectors,-12,addvector,axis=0)
    else:pass

    return vectors


#--------------------------------------------------------------速度調整法-----------------------------------------------------------#

#段階調整法
def velocity_pulse(vectors,t,dt,v_tether,v_tmp,startdays = 0):
    
    #月面操作用
    signs = -1.0 if (t>t_return - startdays*24*3600*1/dt) else 1.0
    t += timer(startdays,dt)
    
    if(t == timer(10.65,dt)):v_tmp = v_tether
    #遷移区間
    if(timer(12.65,dt) <= t <= timer(14.65,dt)):
        v_tether -= 2*v_tmp/(2.0*24*3600) * dt
    else:
        if(t <= timer(2,dt) or t >= timer(25.3,dt)):
            v_tether = 3500.0/(24*3600*27.3217/2)*signs
        else:
            v_tether = 3500.0/(24*3600*27.3217/2)*signs
    '''
    #CW操作
    signs = -1.0 if (t>t_return - startdays*24*3600*1/dt) else 1.0
    t += timer(startdays,dt)
    if(t == timer(10.65,dt)):v_tmp = v_tether
    #遷移区間
    if(timer(12.65,dt) <= t <= timer(14.65,dt)):
        v_tether -= 2*v_tmp/(2.0*24*3600) * dt
    else:
        if(t <= timer(2,dt) or t >= timer(25.3,dt)):
            v_tether = 19000.0/(24*3600*27.3217/2)*signs
        else:
            v_tether = 19000.0/(24*3600*27.3217/2)*signs
    '''
    
    
    return v_tether,v_tmp

    

#相対速度法
def velocity_earth(vectors,t,dt,v_tether):
    if(t <= timer(5,dt) or t >= timer(22.3,dt)):
        v_rate = 8000.0/40551.23
    elif(timer(5,dt) < t < t_return or timer(16,dt) < t < timer(22.3,dt)):
        v_rate = 8000.0/40551.23
    else:
        v_rate = 8000.0/40551.23
    r = (vectors[-1,0]**2 + vectors[-1,1]**2)**(1/2)
    v_tether = (vectors[-1,3]*(vectors[-1,0]/r) + vectors[-1,4]*(vectors[-1,1]/r))*v_rate

    return v_tether

#張力制御法
def velocity_tension(vectors,t,dt,v_tether,v_tmp,dl,startdays=0):
    signs = -1.0 if (t>t_return - startdays*24*3600*1/dt) else 1.0
    if(t == timer(10.65,dt)):v_tmp = v_tether
    #遷移区間
    if(timer(10.65,dt) <= t <= timer(16.65,dt)):
        v_tether -= 2*v_tmp/(6.0*24*3600) * dt
    #それ以外
    else:
        T = (k_ratios[1]*l_0/l_0_short*max((((vectors[2,0]-vectors[1,0])**2 + (vectors[2,1]-vectors[1,1])**2)**(1/2)-l_0_short),0.0))/Areas[0]
        if(T >= sigma_par_p):
            if(signs > 0):
                v_rate = 1-(1.0*10**-5)
            else:
                v_rate = 1+(1.0*10**-5)
        elif(T < sigma_par_m):
            if(signs > 0):
                v_rate = 1+(1.0*10**-5)
            else:
                v_rate = 1-(1.0*10**-5)
        else:
            v_rate = 1.0
        
        v_tether = abs(v_tether) * v_rate * signs
        if(v_tether > v_tether_max): 
            v_tether = v_tether_max
        elif(v_tether < v_tether_min):
            v_tether = v_tether_min
        else:pass

    return v_tether,v_tmp


#--------------------------------------------------------------質点調整法-----------------------------------------------------------#

#月面操作
def change_nodes_moon(vectors,t,dt,counters,startdays=0):
    v_tether = counters[0]
    dl = counters[1]
    counter = counters[2]
    counter_2 = counters[3]

    checker = 1
    if(t>t_return - startdays*24*3600*1/dt):
        checker = -1
    if(checker):
        #伸ばす
        if(checker == 1):
            dl += v_tether * dt 
            #伸びが10kmを超えたら
            if(dl >= l_0_short):
                #新しい質点を追加
                x = vectors[1,0]*dl/(l_0_short+dl)
                y = (vectors[1,1]-1737.0)*dl/(l_0_short+dl) + 1737.0
                z = vectors[1,2]*dl/(l_0_short+dl)
                u = vectors[1,3]*dl/(l_0_short+dl)
                v = vectors[1,4]*dl/(l_0_short+dl)
                w = vectors[1,5]*dl/(l_0_short+dl)
                #追加ベクトル
                addvector = np.array([x,y,z,u,v,w])
                vectors = np.insert(vectors,1,addvector,axis=0)
                counter += 1
                #伸びをリセット
                dl -= l_0_short
                #追加質点数が10個を超えたら
                if(counter >= 10):
                    #追加した21~29を削除，30を代表値として残す
                    vectors = np.delete(vectors,slice(21,30),0)
                    #カウンタのリセット
                    counter = 0
                    #全体の質点数を1増加
                    counter_2 += 1
            else:pass

        #縮む->
        elif(checker == -1):
            dl += v_tether * dt
            #縮みが10kmを超えたら
            if(dl <= -l_0_short):
                #質点を削除
                vectors = np.delete(vectors,1,0)
                counter -= 1
                #縮みをリセット
                dl += l_0_short
                #削除質点が10個超えたら
                if(counter <= -10):
                    #第10,11質点を10等分
                    addvector = np.zeros((9,6))
                    vec1 = vectors[10]
                    vec2 = vectors[11]
                    for i in range(9):
                        x = (vec2[0] - vec1[0])/10.0 * (i + 1) + vec1[0]
                        y = (vec2[1] - vec1[1])/10.0 * (i + 1) + vec1[1]
                        z = (vec2[2] - vec1[2])/10.0 * (i + 1) + vec1[2]
                        u = (vec2[3] - vec1[3])/10.0 * (i + 1) + vec1[3]
                        v = (vec2[4] - vec1[4])/10.0 * (i + 1) + vec1[4]
                        w = (vec2[5] - vec1[5])/10.0 * (i + 1) + vec1[5]
                        addvector[i] = np.array([x,y,z,u,v,w])
                    #質点の追加
                    vectors = np.insert(vectors,11,addvector,axis=0)
                    #カウンタのリセット
                    counter = 0
                    #全体の質点数を1減少
                    counter_2 -= 1
        else:pass
        
        counters = [dl,counter,counter_2]

    return vectors,counters

#CW操作(二次元のまま！)
def change_nodes_cw(vectors,t,dt,counters,startdays=0):
    v_tether = counters[0]
    dl = counters[1]
    counter = counters[2]
    counter_2 = counters[3]

    checker = 1

    if(t>t_return - startdays*24*3600*1/dt):
        checker = -1
    if(checker):
        #伸ばす
        if(checker == 1):
            dl += v_tether * dt 
            #伸びが10kmを超えたら
            if(dl >= l_0_short):
                #新しい質点を追加
                #N-2->CWベクトルを内分
                addvector = (vectors[-3]-vectors[-2])*dl/(dl+l_0_short)+vectors[-2]
                vectors = np.insert(vectors,-2,addvector,axis=0)
                counter += 1
                #伸びをリセット
                dl -= l_0_short
                #追加質点数が10個を超えたら
                if(counter >= 10):
                    #追加したN-22~N-30を削除，N-31を代表値として残す
                    temp = vectors.shape[0]
                    vectors = np.delete(vectors,slice(temp-31,temp-22),0)
                    #カウンタのリセット
                    counter = 0
                    #全体の質点数を1増加
                    counter_2 += 1
            else:pass

        #縮む->
        elif(checker == -1):
            dl += v_tether * dt
            #縮みが10kmを超えたら
            if(dl <= -l_0_short):
                #質点を削除
                vectors = np.delete(vectors,-3,0)
                counter -= 1
                #縮みをリセット
                dl += l_0_short
                #削除質点が10個超えたら
                if(counter <= -10):
                    #第N-13,N-12質点を10等分
                    addvector = np.zeros((9,6))
                    vec1 = vectors[-13]
                    vec2 = vectors[-12]
                    for i in range(9):
                        addvector[i] = (vec2-vec1)*(i+1)/10 + vec1
                    #質点の追加
                    vectors = np.insert(vectors,-12,addvector,axis=0)
                    #カウンタのリセット
                    counter = 0
                    #全体の質点数を1減少
                    counter_2 -= 1
            else:pass
        
        counters = [dl,counter,counter_2]
    return vectors,counters

#--------------------------------------------------------運動方程式_質点判定式-------------------------------------------------------#
#月面操作
@jit(f8[:](i8,i8,i8,f8,i8))
def allocate_par_moon(counter,counter_2,i,dl,code1):
    #返り値
    ret_l_0 = l_0_short
    k_m = 0.0
    k_p = 0.0
    A_point = 0.0        

    #追加質点がゼロ -> counter分だけが伸展テザー
    if(counter_2 == 0):
        #質点1,2を10分割したときのばね定数*10*2を計算
        k_node1 = np.array([(k_ratios[1]-k_ratios[0])/10*i + k_ratios[0] for i in range(10)]) * l_0/l_0_short
        k_node2 = np.array([(k_ratios[2]-k_ratios[1])/10*i + k_ratios[1] for i in range(10)]) * l_0/l_0_short
        A_node1 = np.array([(Areas[1]-Areas[0])/10*i + Areas[0] for i in range(10)])
        A_node2 = np.array([(Areas[2]-Areas[1])/10*i + Areas[1] for i in range(11)])

        k_divnodes = np.hstack((k_node1,k_node2))
        A_divnodes = np.hstack((A_node1,A_node2))

        #伸展がない
        if(counter == 0):
            #第1質点(操作質点)
            if(i == 1):
                code1 = 1
                ret_l_0 = l_0_short + dl                             #自然長(伸びも考える)
                ret_l_0 = max(ret_l_0,l_0_min)                           #最短値を設定(張力の発散を抑制)
                k_m = k_divnodes[0]* l_0_short/(l_0_short+dl)        #月方向のばね定数
                k_p = k_divnodes[1]                                  #地球方向のばね定数
                A_point = A_divnodes[1]                              #断面積
            
            #通常ノードとの境目
            elif(i == 20):
                code1 = 2
                k_m = k_divnodes[19]
                k_p = k_ratios[2]
                A_point = A_divnodes[20]
            
            #基礎ノードの分割区間   常に20個存在
            elif(1 < i < 20):
               k_m = k_divnodes[i - 1]
               k_p = k_divnodes[i]
               A_point = A_divnodes[i]

            #通常質点
            else:
                ret_l_0 = l_0
                k_m = k_ratios[i - 19]                 #月方向のばね定数
                k_p = k_ratios[i - 18]                 #地球方向のばね定数
                A_point = Areas[i - 18]
                
        #伸展がある
        else:
            #第1質点　伸展テザーの要素になる
            if(i == 1):
                code1 = 1
                ret_l_0 = l_0_short + dl                         #自然長(伸びも考える)
                ret_l_0 = max(ret_l_0, l_0_min)                      #最短値を設定(張力の発散を抑制)
                k_m = k_ratios[0]* l_0/(l_0_short+dl)            #月方向のばね定数
                k_p = k_ratios[0]* l_0/ l_0_short                #地球方向のばね定数
                A_point = Areas[0]
            
            #通常質点との境目
            elif(i == 20 + counter):
                code1 = 2
                k_m = k_divnodes[i - counter]
                k_p = k_ratios[2]
                A_point = A_divnodes[i - counter]

            #伸展テザー部分
            elif(1 < i <= counter):
                k_m = k_ratios[0]* l_0/ l_0_short                 #月方向のばね定数
                k_p = k_ratios[0]* l_0/ l_0_short                 #地球方向のばね定数
                A_point = A_divnodes[0]

            #基礎テザーの分割部分　20個存在
            elif(counter < i < 20 + counter):
                k_m = k_divnodes[i - counter - 1]
                k_p = k_divnodes[i - counter]
                A_point = A_divnodes[i - counter]

            #通常質点
            else:
                ret_l_0 = l_0
                k_m = k_ratios[i - 19 - counter]                 #月方向のばね定数
                k_p = k_ratios[i - 18 - counter]                 #地球方向のばね定数
                A_point = Areas[i - 18 - counter]
            
    #ノード1分割部分を考える必要がある
    elif(counter_2 == 1):
        #質点1を10分割したときのばね定数*10を計算
        k_node1 = np.array([(k_ratios[1]-k_ratios[0])/10*i + k_ratios[0] for i in range(10)]) * l_0/ l_0_short
        A_divnodes = np.array([(Areas[1]-Areas[0])/10*i + Areas[0] for i in range(11)])

        #第一質点(操作質点) ここは完全に伸展テザー
        if(i == 1):
            code1 = 1
            ret_l_0 = l_0_short + dl                             #自然長(伸びも考える)
            ret_l_0 = max(ret_l_0, l_0_min)                           #最短値を設定(張力の発散を抑制)
            k_m = k_ratios[0]* l_0/(l_0_short+dl)            #月方向のばね定数
            k_p = k_ratios[0]* l_0/ l_0_short                 #地球方向のばね定数
            A_point = Areas[0]                              

        #通常部との境目
        elif(i == counter + 20):
            k_m = k_node1[9]                #月方向のばね定数
            k_p = k_ratios[1]               #地球方向のばね定数
            A_point = A_divnodes[10]           #質量
                
        #分割区間その１(伸展テザー)　質点2~counter + 10
        elif(1 < i <= counter + 10):
            k_m = k_ratios[0]* l_0/ l_0_short                 #月方向のばね定数
            k_p = k_ratios[0]* l_0/ l_0_short                 #地球方向のばね定数
            A_point = Areas[0]                                #質量                
                
        #分割区間その２(ノード1分割区間) 質点は10個で確定
        elif(counter + 10 < i <= counter + 20):
            k_m = k_node1[i - counter - 11]                 #月方向のばね定数
            k_p = k_node1[i - counter - 10]                 #地球方向のばね定数
            A_point = A_divnodes[i - counter - 10]             #質量
                
        #通常区間
        else:
            ret_l_0 =  l_0                                    #自然長(100km基準)
            k_m = k_ratios[i-(18 + counter + counter_2) - 1]      #月側のばね定数
            k_p = k_ratios[i-(18 + counter + counter_2)]  #地球側のばね定数
            A_point = Areas[i-(18 + counter + counter_2)]     #質量

            
    #通常テザー部が分割部よりも長く伸展
    elif(counter_2 >= 2):
        #第一質点(操作質点) この場合確実に伸展テザー区間
        if(i == 1):
            code1 = 1
            ret_l_0 =  l_0_short + dl                         #自然長(伸びも考える)
            ret_l_0 = max(ret_l_0, l_0_min)                   #最短値を設定(張力の発散を抑制)
            k_m = k_ratios[0]* l_0/( l_0_short+dl)            #月方向のばね定数
            k_p = k_ratios[0]* l_0/ l_0_short                 #地球方向のばね定数
            A_point = Areas[0]                                #質量
        
        #通常部との境目
        elif(i == counter + 20):
            code1 = 2
            k_m = k_ratios[0]* l_0/ l_0_short           #月方向のばね定数
            k_p = k_ratios[0]                           #地球方向のばね定数
            A_point = Areas[0]                          #質量
                    
        #分割区間 すべて伸展部 2~counter+20
        elif(1 < i < counter + 20):
            k_m = k_ratios[0]* l_0/ l_0_short           #月方向のばね定数
            k_p = k_ratios[0]* l_0/ l_0_short           #地球方向のばね定数
            A_point = Areas[0]                          #質量
                
        #追加質点部
        elif(counter + 20 < i < counter + counter_2 + 20):
            ret_l_0 =  l_0
            k_m = k_ratios[0]           #月方向のばね定数
            k_p = k_ratios[0]           #地球方向のばね定数
            A_point = Areas[0]         #質量
                
        #通常質点
        else:
            ret_l_0 =  l_0                                     #自然長(100km基準)
            k_m = k_ratios[i-(18 + counter + counter_2) - 1]   #月側のばね定数
            k_p = k_ratios[i-(18 + counter + counter_2)]       #地球側のばね定数
            A_point = Areas[i-(18 + counter + counter_2)]      #質量
    
    return np.array([ret_l_0,k_m,k_p,A_point,code1])

#CW操作
@jit(f8[:](i8,i8,i8,f8,i8,i8))
def allocate_par_cw(counter,counter_2,i,dl,code1,N_earth):
    #返り値
    ret_l_0 = l_0_short
    k_m = 0.0
    k_p = 0.0
    A_point = 0.0     

    #追加質点がゼロ -> counter分だけが伸展テザー
    if(counter_2 == 0):
        #質点1,2を10分割したときのばね定数*10*2を計算
        k_node1 = np.array([(k_ratios[-2]-k_ratios[-3])/10*i + k_ratios[-3] for i in range(10)]) * l_0/ l_0_short
        k_node2 = np.array([(k_ratios[-3]-k_ratios[-4])/10*i + k_ratios[-4] for i in range(10)]) * l_0/ l_0_short
        A_node1 = np.array([(Areas[-2]-Areas[-3])/10*i + Areas[-3] for i in range(11)])
        A_node2 = np.array([(Areas[-3]-Areas[-4])/10*i + Areas[-4] for i in range(10)])

        k_divnodes = np.hstack((k_node2,k_node1))
        A_divnodes = np.hstack((A_node2,A_node1))

        #伸展がない
        if(counter == 0):
            #第N_earth-3質点(操作質点)
            if(i == N_earth-3):
                code1 = 1
                ret_l_0 = l_0_short + dl                       #自然長(伸びも考える)
                ret_l_0 = max(ret_l_0,l_0_min)                 #最短値を設定(張力の発散を抑制)
                k_m = k_node1[8]                               #月方向のばね定数
                k_p = k_node1[9]* l_0_short/(l_0_short+dl)     #地球方向のばね定数
                A_point = A_divnodes[0]                       
            
            #通常質点との境目
            elif(i == N_earth - 22):
                code1 = 2
                k_m = k_ratios[-5]
                k_p = k_divnodes[0]
                A_point = A_divnodes[0]
            
            #基礎ノードの分割区間
            elif(N_earth - 22 < i < N_earth - 3):
                k_m = k_divnodes[i -(N_earth - 22) - 1]
                k_p = k_divnodes[i -(N_earth - 22)]
                A_point = A_divnodes[i - (N_earth - 22)]

            #通常質点
            else:
                ret_l_0 =  l_0
                k_m = k_ratios[i - 1]             #月方向のばね定数
                k_p = k_ratios[i]                 #地球方向のばね定数
                A_point = Areas[i]               
                
        #伸展がある
        else:
            #第1質点　伸展テザーの要素になる
            if(i == N_earth-3):
                code1 = 1
                ret_l_0 =  l_0_short + dl                 #自然長(伸びも考える)
                ret_l_0 = max(ret_l_0,l_0_min)            #最短値を設定(張力の発散を抑制)
                k_m = k_ratios[-2]* l_0/l_0_short         #月方向のばね定数
                k_p = k_ratios[-2]* l_0/(l_0_short+dl)    #地球方向のばね定数
                A_point = Areas[-2]                       #質量

            #通常テザーとの境目
            elif(i == N_earth - 22 - counter):
                code1 = 2
                k_m = k_ratios[-5]
                k_p = k_divnodes[0]
                A_point = A_divnodes[0]

            #伸展テザー部分
            elif(N_earth-3-counter <= i < N_earth-3):
                k_m = k_ratios[-2]* l_0/ l_0_short        #月方向のばね定数
                k_p = k_ratios[-2]* l_0/ l_0_short        #地球方向のばね定数
                A_point = Areas[-2]                       #質量

            #基礎テザー分割部分
            elif(N_earth - 22 - counter < i < N_earth - 3 - counter):
                k_m = k_divnodes[i - (N_earth - 12 - counter) - 1]              #月方向のばね定数
                k_p = k_divnodes[i - (N_earth - 12 - counter)]                  #地球方向のばね定数
                A_point = A_divnodes[i - (N_earth - 12 - counter)]              #質量
                    
            #通常質点
            else:
                ret_l_0 =  l_0
                k_m = k_ratios[i - 1]                 #月方向のばね定数
                k_p = k_ratios[i]                     #地球方向のばね定数
                A_point = Areas[i]                   
            
    #ノード1分割部分を考える必要がある
    elif(counter_2 == 1):
        #質点1を10分割したときのばね定数*10を計算
        k_node1 = np.array([(k_ratios[N_earth-2]-k_ratios[N_earth-3])/10*i + k_ratios[N_earth-3] for i in range(10)]) *  l_0/ l_0_short
        A_node1 = np.array([(Areas[N_earth-2]-Areas[N_earth-3])/10*i + Areas[N_earth-3] for i in range(11)])

        #第一質点(操作質点) ここは完全に伸展テザー
        if(i == 1):
            code1 = 1
            ret_l_0 =  l_0_short + dl                  #自然長(伸びも考える)
            ret_l_0 = max(ret_l_0, l_0_min)            #最短値を設定(張力の発散を抑制)
            k_m = k_ratios[-2]* l_0/ l_0_short         #月方向のばね定数
            k_p = k_ratios[-2]* l_0/( l_0_short+dl)    #地球方向のばね定数
            A_point = Areas[-2]                        #質量

        #通常質点との境目
        elif(i == N_earth - 22 - counter):
            code1 = 2
            k_m = k_ratios[-4]
            k_p = k_node1[0]
            A_point = A_node1[0]
            
        #分割区間その１(伸展テザー)　質点N-3-(counter+10)~N-3
        elif(N_earth-3-(counter+10) <= i < N_earth-3):
            k_m = k_ratios[-2]* l_0/ l_0_short         #月方向のばね定数
            k_p = k_ratios[-2]* l_0/ l_0_short         #地球方向のばね定数
            A_point = Areas[-2]                      
                
        #分割区間その２(ノード1分割区間) 質点は10個で確定
        elif(N_earth - 22 - counter < i < N_earth-3-(counter+10)):
            k_m = k_node1[i - (N_earth - 22 - counter)-1]               #月方向のばね定数
            k_p = k_node1[i - (N_earth - 22 - counter)]                 #地球方向のばね定数
            A_point = A_node1[i - (N_earth - 22 - counter)]             #質量
                
        #通常区間
        else:
            ret_l_0 =  l_0                                    #自然長(100km基準)
            k_m = k_ratios[i-1]      #月側のばね定数
            k_p = k_ratios[i]  #地球側のばね定数
            A_point = Areas[i]    #質量
            
    #通常テザー部分割部からちょうど抜ける
    elif(counter_2 == 2):
        #第一質点(操作質点) この場合確実に伸展テザー区間
        if(i == N_earth-3):
            code1 = 1
            ret_l_0 =  l_0_short + dl                         #自然長(伸びも考える)
            ret_l_0 = max(ret_l_0, l_0_min)                   #最短値を設定(張力の発散を抑制)
            k_m = k_ratios[-2]* l_0/(l_0_short+dl)            #月方向のばね定数
            k_p = k_ratios[-2]* l_0/l_0_short                 #地球方向のばね定数
            A_point = Areas[-2]                               #質量
        
        #通常質点との境目
        elif(i == N_earth-22-counter):
            code1 = 2
            k_m = k_ratios[-3]
            k_p = k_node1[0]
            A_point = A_node1[0]
                    
        #分割区間 すべて伸展部 2~counter+20
        elif(N_earth - 22 - counter <= i < N_earth-3):
            k_m = k_ratios[-2]* l_0/ l_0_short   #月方向のばね定数
            k_p = k_ratios[-2]* l_0/ l_0_short   #地球方向のばね定数
            A_point = Areas[-2]                #質量
                
        #通常質点
        else:
            ret_l_0 =  l_0                          #自然長(100km基準)
            k_m = k_ratios[i-1]                     #月側のばね定数
            k_p = k_ratios[i]                       #地球側のばね定数
            A_point = Areas[i]    

    #分割部が完全に伸展テザーに置き換わる
    elif(counter_2 >= 3):
        #第一質点(操作質点) この場合確実に伸展テザー区間
        if(i == N_earth-3):
            code1 = 1
            ret_l_0 =  l_0_short + dl                             #自然長(伸びも考える)
            ret_l_0 = max(ret_l_0, l_0_min)                           #最短値を設定(張力の発散を抑制)
            k_m = k_ratios[-2]* l_0/(l_0_short+dl)            #月方向のばね定数
            k_p = k_ratios[-2]* l_0/l_0_short                 #地球方向のばね定数
            A_point = Areas[-2]                                #質量
        
        #通常質点との境目
        elif(i == N_earth-22-counter):
            code1 = 2
            k_m = k_ratios[-2]
            k_p = k_node1[0]
            A_point = A_node1[0]
        
        #伸展部と基礎テザーとの境目
        elif(i == N_earth - 22 - counter - counter_2):
            ret_l_0 = l_0
            k_m = k_ratios[-3]
            k_p = k_ratios[-2]
            A_point = Areas[-2]
        
        #分割区間 すべて伸展部 2~counter+20
        elif(N_earth - 22 - counter < i < N_earth-3):
            k_m = k_ratios[-2]* l_0/l_0_short   #月方向のばね定数
            k_p = k_ratios[-2]* l_0/l_0_short   #地球方向のばね定数
            A_point = Areas[-2]           

        #追加質点
        elif(N_earth - 22 - counter - counter_2 < i < N_earth - 22 - counter):
            ret_l_0 = l_0      #自然長
            k_m = k_ratios[-2]           #月方向のばね定数
            k_p = k_ratios[-2]           #地球方向のばね定数
            A_point = Areas[-2]          #質量
                
        #通常質点
        else:
            ret_l_0 = l_0                                    #自然長(100km基準)
            k_m = k_ratios[i-1]      #月側のばね定数
            k_p = k_ratios[i]        #地球側のばね定数
            A_point = Areas[i]      #質量

    return np.array([ret_l_0,k_m,k_p,A_point,code1])

@jit(f8[:](i8,i8,i8,f8,i8))
def test_nodes(counter,counter_2,i,dl,code1):

    ret_l_0 = 0.0
    k_m = 0.0
    k_p = 0.0
    m_point = 0.0
    A_point = 0.0
    code1 = 0
    if(i == 1):
        ret_l_0 = l_0_short + dl #自然長(伸びも考える)
        k_m = k_ratios[0]*l_0/l_0  #月方向のバネ定数
        k_p = k_ratios[1]*l_0/l_0_short #地球方向のばね定数
        m_point = masses[0]/10.0 #質量
        A_point = Areas[0]
        damp = 2*(m_point*k_p)**(1/2)*cr #減衰係数
        code1 = 1 
        #質点間隔10km部分
        #maxで30個，minで10個
    elif(i < 20 + counter):
        ret_l_0 = l_0_short
        k_m = k_ratios[1]*l_0/l_0_short
        k_p = k_m
        m_point = masses[0]/(10.0 + counter)
        A_point = Areas[0]
        damp = 2*(m_point*k_p)**(1/2)*cr
    #境目
    elif(i == 20 + counter):
        ret_l_0 = l_0_short
        k_m = k_ratios[1]*l_0/l_0_short
        k_p = k_ratios[2]
        m_point = masses[0]/10.0
        A_point = Areas[0]
        damp = 2*(m_point*k_p)**(1/2)*cr
        code1 = 2    
    #そのた
    elif(counter_2 != 0 and i > 20 + counter and i <= 20 + counter + counter_2 ):
        ret_l_0 = l_0
        k_m = k_ratios[2]
        k_p = k_ratios[3]
        m_point = masses[2]
        A_point = Areas[2]
        damp = 2*(m_point*k_p)**(1/2)*cr
    else:
        ret_l_0 = l_0
        k_m = k_ratios[i-(18+counter+ counter_2)]
        k_p = k_ratios[i-(18+counter+ counter_2)+1]
        m_point = masses[i-(18+counter+ counter_2)]
        A_point = Areas[i-(18+counter+ counter_2)]
        damp = 2*(m_point*k_p)**(1/2)*cr
    
    return np.array([ret_l_0,k_m,k_p,A_point,code1])

