from . import values as vl
import time
import os
import numpy as np
from numba import jit, f8, i8, b1, void

#外力計算
@jit(f8[:,:](f8[:,:],f8[:],f8[:],f8[:],f8[:],f8[:,:],i8,i8,f8,i8))
def Forces(vectors,spring,mass,damps,Areas,climbers,counter,counter_2,dl,case):
    #パラメータの準備
    #地球座標番号の取得
    N_earth = vectors.shape[0] #質点数
    ret_Forces = np.zeros((N_earth,30)) #返り値用のベクトル
    R = (vectors[N_earth-1,0]**2 + vectors[N_earth-1,1]**2 + vectors[N_earth-1,2]**2)**(3/2) #月-地球の距離
    Firr = -vl.myu/R*vectors[N_earth-1,:3] #慣性力
    '''
    omega = vl.alpha/R**(2/3) #角速度
    if(vectors[N_earth-1,3] > 0):
        alpha = -2*vl.alpha*(vectors[N_earth-1,2]**2 + vectors[N_earth-1,3]**2)**(1/2)/R
    else:
        alpha = 2*vl.alpha*(vectors[N_earth-1,2]**2 + vectors[N_earth-1,3]**2)**(1/2)/R
    '''

    Mcw = vl.Mcw
    spring_base = 0.0
    Areas_base = 0.0
    k_node1 = np.zeros(10)
    k_node2 = np.zeros(10)
    A_node1 = np.zeros(11)
    A_node2 = np.zeros(10)

    k_divnodes = np.hstack((k_node1,k_node2))
    A_divnodes = np.hstack((A_node1,A_node2))

    #伸展/回収したテザーの質量だけCW質量も変化．(CW操作の場合)
    #dl + counter + counter2,操作テザーの質量は質点N-3依存とする
    if(case == 0):
        spring_base = spring[0]
        Areas_base = Areas[0]
        k_node1 = np.array([(spring[1]-spring[0])/10*i + spring[0] for i in range(10)]) * vl.l_0/vl.l_0_short
        k_node2 = np.array([(spring[2]-spring[1])/10*i + spring[1] for i in range(10)]) * vl.l_0/vl.l_0_short
        A_node1 = np.array([(Areas[1]-Areas[0])/10*i + Areas[0] for i in range(11)])
        A_node2 = np.array([(Areas[2]-Areas[1])/10*(i+1) + Areas[1] for i in range(10)])

        k_divnodes = np.hstack((k_node1,k_node2))
        A_divnodes = np.hstack((A_node1,A_node2))
        
    elif(case == 1):
        Mcw = vl.Mcw - (counter*mass[-3]/10.0 + counter_2*mass[-3] + mass[-3]*dl/vl.l_0)
        spring_base = spring[-2]
        Areas_base = Areas[-2]
        k_node1 = np.array([(spring[-3]-spring[-4])/10*i + spring[-4] for i in range(10)]) * vl.l_0/vl.l_0_short
        k_node2 = np.array([(spring[-2]-spring[-3])/10*i + spring[-3] for i in range(11)]) * vl.l_0/vl.l_0_short
        A_node1 = np.array([(Areas[-3]-Areas[-4])/10*i + Areas[-4] for i in range(10)])
        A_node2 = np.array([(Areas[-2]-Areas[-3])/10*i + Areas[-3] for i in range(11)])

        k_divnodes = np.hstack((k_node1,k_node2))
        A_divnodes = np.hstack((A_node1,A_node2))

    
    #1~N-1Node
    for i in range(1,N_earth-2):
        #使いまわすパラメータの初期化
        l_0 = vl.l_0_short
        k_m = 0.0
        k_p = 0.0
        A_point = 0.0 
        code1 = 0
        #質点番号ごとの操作(分割区間の自然長/質量/ばね定数)
        ##############################################################月面操作の場合######################################################################
        if(case == 0):
            #通常テザー部が分割部よりも長く伸展
            if(counter_2 >= 2):
                #通常質点
                if(i > counter + counter_2 + 20):
                    l_0 = vl.l_0                                        #自然長(100km基準)
                    k_m = spring[i-(18 + counter + counter_2) - 1]      #月側のばね定数
                    k_p = spring[i-(18 + counter + counter_2)]          #地球側のばね定数
                    A_point = Areas[i-(18 + counter + counter_2)]       #断面積
                    
                #分割区間 すべて伸展部 2~counter+20
                elif(1 < i < counter + 20):
                    k_m = spring_base* vl.l_0/ vl.l_0_short             #月方向のばね定数
                    k_p = spring_base* vl.l_0/ vl.l_0_short             #地球方向のばね定数
                    A_point = Areas_base                                #断面積
                            
                #追加質点部
                elif(counter + 20 < i <= counter + counter_2 + 20):
                    l_0 = vl.l_0
                    k_m = spring_base                                   #月方向のばね定数
                    k_p = spring_base                                   #地球方向のばね定数
                    A_point = Areas_base                                #断面積

                #第一質点(操作質点) この場合確実に伸展テザー区間
                elif(i == 1):
                    code1 = 1
                    l_0 = vl.l_0_short + dl                             #自然長(伸びも考える)
                    #l_0 = max(l_0,vl.l_0_min)                           #最短値を設定(張力の発散を抑制)
                    k_m = min(spring_base*vl.l_0/l_0,spring_base*vl.l_0/vl.l_0_short * vl.k_max_rate)     #月方向のばね定数
                    k_p = spring_base* vl.l_0/vl.l_0_short              #地球方向のばね定数
                    A_point = Areas_base                                #断面積
                    
                #通常部との境目
                elif(i == counter + 20):
                    code1 = 2
                    k_m = spring_base* vl.l_0/ vl.l_0_short             #月方向のばね定数
                    k_p = spring_base                                   #地球方向のばね定数
                    A_point = Areas_base                                #断面積
        
            #追加質点がゼロ -> counter分だけが伸展テザー
            elif(counter_2 == 0):
                #質点1,2を10分割したときのばね定数*10*2を計算
                #伸展がない
                if(counter == 0):
                    #通常質点
                    if(i > 20):
                        l_0 = vl.l_0
                        k_m = spring[i - 19]                            #月方向のばね定数
                        k_p = spring[i - 18]                            #地球方向のばね定数
                        A_point = Areas[i - 18]
                    
                    #基礎ノードの分割区間   常に20個存在
                    elif(1 < i < 20):
                        k_m = k_divnodes[i - 1]
                        k_p = k_divnodes[i]
                        A_point = A_divnodes[i]

                    #第1質点(操作質点)
                    elif(i == 1):
                        code1 = 1
                        l_0 = vl.l_0_short + dl                         #自然長(伸びも考える)
                        #l_0 = max(l_0,vl.l_0_min)                      #最短値を設定(張力の発散を抑制)
                        k_m = min(k_divnodes[0]*vl.l_0_short/l_0,k_divnodes[0]*vl.k_max_rate)            #月方向のばね定数
                        k_p = k_divnodes[1]                             #地球方向のばね定数
                        A_point = A_divnodes[1]                         #断面積
            
                    #通常ノードとの境目
                    elif(i == 20):
                        code1 = 2
                        k_m = k_divnodes[19]
                        k_p = spring[2]
                        A_point = A_divnodes[20]
                        
                #伸展がある
                else:
                    #通常質点
                    if(i > 20+counter):
                        l_0 = vl.l_0
                        k_m = spring[i - 19 - counter]                 #月方向のばね定数
                        k_p = spring[i - 18 - counter]                 #地球方向のばね定数
                        A_point = Areas[i - 18 - counter]
                    
                    #伸展テザー部分
                    elif(1 < i <= counter):
                        k_m = spring_base* vl.l_0/ vl.l_0_short        #月方向のばね定数
                        k_p = spring_base* vl.l_0/ vl.l_0_short        #地球方向のばね定数
                        A_point = A_divnodes[0]
                    
                    #基礎テザーの分割部分　20個存在
                    elif(counter < i < 20 + counter):
                        k_m = k_divnodes[i - counter - 1]
                        k_p = k_divnodes[i - counter]
                        A_point = A_divnodes[i - counter]

                    #第1質点　伸展テザーの要素になる
                    elif(i == 1):
                        code1 = 1
                        l_0 = vl.l_0_short + dl                         #自然長(伸びも考える)
                        #l_0 = max(l_0, vl.l_0_min)                      #最短値を設定(張力の発散を抑制)
                        k_m = min(spring_base* vl.l_0/l_0,spring_base* vl.l_0/vl.l_0_short*vl.k_max_rate)                  #月方向のばね定数
                        k_p = spring_base* vl.l_0/vl.l_0_short          #地球方向のばね定数
                        A_point = Areas_base
                    
                    #通常質点との境目
                    elif(i == 20 + counter):
                        code1 = 2
                        k_m = k_divnodes[19]
                        k_p = spring[2]
                        A_point = A_divnodes[20]
      
            #ノード1分割部分を考える必要がある
            elif(counter_2 == 1):
                #質点1を10分割したときのばね定数*10を計算
                A_divnodes = A_node1
                #通常質点
                if(i > counter + 20):
                    l_0 = vl.l_0                                        #自然長(100km基準)
                    k_m = spring[i-(18 + counter + counter_2) - 1]      #月側のばね定数
                    k_p = spring[i-(18 + counter + counter_2)]          #地球側のばね定数
                    A_point = Areas[i-(18 + counter + counter_2)]       #断面積
                
                #分割区間その１(伸展テザー)　質点2~counter + 10
                elif(1 < i <= counter + 10):
                    k_m = spring_base* vl.l_0/ vl.l_0_short             #月方向のばね定数
                    k_p = spring_base* vl.l_0/ vl.l_0_short             #地球方向のばね定数
                    A_point = A_divnodes[0]             
                        
                #分割区間その２(ノード1分割区間) 質点は10個で確定
                elif(counter + 10 < i < counter + 20):
                    k_m = k_node1[i - counter - 11]                     #月方向のばね定数
                    k_p = k_node1[i - counter - 10]                     #地球方向のばね定数
                    A_point = A_divnodes[i - counter - 10]              #断面積

                #第一質点(操作質点) ここは完全に伸展テザー
                elif(i == 1):
                    code1 = 1
                    l_0 = vl.l_0_short + dl                             #自然長(伸びも考える)
                    #l_0 = max(l_0, vl.l_0_min)                          #最短値を設定(張力の発散を抑制)
                    k_m = min(spring_base* vl.l_0/l_0,spring_base*vl.l_0/vl.l_0_short*vl.k_max_rate)                       #月方向のばね定数
                    k_p = spring_base* vl.l_0/vl.l_0_short              #地球方向のばね定数
                    A_point = Areas_base                              

                #通常部との境目
                elif(i == counter + 20):
                    code1  = 2
                    k_m = k_node1[9]                                    #月方向のばね定数
                    k_p = spring[1]                                     #地球方向のばね定数
                    A_point = A_divnodes[10]                            #断面積

        #######################################################################CW操作の場合#########################################################################
        elif(case == 1):
            l_0 = vl.l_0_short
            k_m = 0.0
            k_p = 0.0
            A_point = 0.0 
            code1 = 0
            #通常テザー部が分割部よりも長く伸展
            if(counter_2 >= 3):
                #通常質点
                if(i <= N_earth - 22 - counter - counter_2 + 2):
                    l_0 =  vl.l_0                                       #自然長(100km基準)
                    k_m = spring[i - 1]                                 #月側のばね定数
                    k_p = spring[i]                                     #地球側のばね定数
                    A_point = Areas[i]                                  #断面積
                        
                #分割区間 すべて伸展部 2~counter+20
                elif(N_earth - 22 - counter < i < N_earth - 3):
                    k_m = spring_base* vl.l_0/ vl.l_0_short             #月方向のばね定数
                    k_p = spring_base* vl.l_0/ vl.l_0_short             #地球方向のばね定数
                    A_point = Areas_base                                #断面積
                                
                #追加質点部
                elif(N_earth - 22 - counter - counter_2 + 2 < i < N_earth - 22 - counter):
                    l_0 = vl.l_0
                    k_m = spring_base                                   #月方向のばね定数
                    k_p = spring_base                                   #地球方向のばね定数
                    A_point = Areas_base                                #断面積

                #第一質点(操作質点) この場合確実に伸展テザー区間
                elif(i == N_earth - 3):
                    code1 = 1
                    l_0 = vl.l_0_short + dl                             #自然長(伸びも考える)
                    #l_0 = max(l_0,vl.l_0_min)                           #最短値を設定(張力の発散を抑制)
                    k_m = spring_base* vl.l_0/vl.l_0_short              #月方向のばね定数
                    k_p = min(spring_base*vl.l_0/l_0,spring_base*vl.l_0/vl.l_0_short*vl.k_max_rate)                       #地球方向のばね定数
                    l_0 = vl.l_0_short
                    A_point = Areas_base                                #断面積
                        
                #通常部との境目
                elif(i == N_earth - 22 - counter):
                    code1 = 2
                    l_0 = vl.l_0
                    k_m = spring_base                                   #月方向のばね定数
                    k_p = spring_base* vl.l_0/ vl.l_0_short             #地球方向のばね定数
                    A_point = Areas_base                                #断面積
            
            #追加質点がゼロ -> counter分だけが伸展テザー
            elif(counter_2 == 0):
                #伸展がない
                if(counter == 0):
                    #通常質点
                    if(i < N_earth - 22):
                        l_0 = vl.l_0
                        k_m = spring[i - 1]                             #月方向のばね定数
                        k_p = spring[i]                                 #地球方向のばね定数
                        A_point = Areas[i]                              #断面積
                        
                    #基礎ノードの分割区間   常に20個存在
                    elif(N_earth - 22 < i < N_earth - 3):
                        k_m = k_divnodes[i - (N_earth - 22) - 1]        #月方向のばね定数
                        k_p = k_divnodes[i - (N_earth - 22)]            #地球方向のばね定数
                        A_point = A_divnodes[i - (N_earth - 22)]        #断面積

                    #第1質点(操作質点)
                    elif(i == N_earth - 3):
                        code1 = 1
                        l_0 = vl.l_0_short + dl                         #自然長(伸びも考える)
                        #l_0 = max(l_0,vl.l_0_min)                       #最短値を設定(張力の発散を抑制)
                        k_m = k_divnodes[18]                            #月方向のばね定数
                        k_p = min(k_divnodes[19]*vl.l_0_short/l_0,k_divnodes[19]*vl.k_max_rate)           #地球方向のばね定数
                        l_0 = vl.l_0_short
                        A_point = A_divnodes[19]                        #断面積
                
                    #通常ノードとの境目
                    elif(i == N_earth - 22):
                        code1 = 2
                        l_0 = vl.l_0
                        k_m = spring[-5]                                #月方向のばね定数                         
                        k_p = k_divnodes[0]                             #地球方向のばね定数
                        A_point = A_divnodes[0]                         #断面積
                            
                #伸展がある
                else:
                    #通常質点
                    if(i < N_earth - 22 - counter):
                        l_0 = vl.l_0
                        k_m = spring[i - 1]                             #月方向のばね定数
                        k_p = spring[i]                                 #地球方向のばね定数
                        A_point = Areas[i]                              #断面積
                        
                    #伸展テザー部分
                    elif(N_earth - 3 - counter + 1 < i < N_earth - 3):
                        k_m = spring_base* vl.l_0/ vl.l_0_short         #月方向のばね定数
                        k_p = spring_base* vl.l_0/ vl.l_0_short         #地球方向のばね定数
                        A_point = Areas_base                            #断面積

                    #第1質点　伸展テザーの要素になる
                    elif(i == N_earth - 3):
                        code1 = 1
                        l_0 = vl.l_0_short + dl                         #自然長(伸びも考える)
                        #l_0 = max(l_0, vl.l_0_min)                      #最短値を設定(張力の発散を抑制)
                        k_m = spring_base* vl.l_0/vl.l_0_short          #月方向のばね定数
                        k_p = min(spring_base* vl.l_0/l_0,spring_base* vl.l_0/vl.l_0_short*vl.k_max_rate)                   #地球方向のばね定数
                        l_0 = vl.l_0_short
                        A_point = Areas_base
                        
                    #基礎テザーの分割部分　20個存在 + 1
                    elif(N_earth - 22 - counter < i <= N_earth - 3 - counter + 1):
                        k_m = k_divnodes[i - (N_earth - 22 - counter) - 1]
                        k_p = k_divnodes[i - (N_earth - 22 - counter)]
                        A_point = A_divnodes[i - (N_earth - 22 - counter)]
                        
                    #通常質点との境目
                    elif(i == N_earth - 22 - counter):
                        code1 = 2
                        l_0 = vl.l_0
                        k_m = spring[-5]
                        k_p = k_divnodes[0]
                        A_point = A_divnodes[0]
        
            #ノード1分割部分を考える必要がある
            elif(counter_2 == 1):
                #質点1を10分割したときのばね定数*10を計算
                A_divnodes = A_node2
                #通常質点
                if(i < N_earth - 22 - counter):
                    l_0 = vl.l_0                                        #自然長(100km基準)
                    k_m = spring[i-1]                                   #月側のばね定数
                    k_p = spring[i]                                     #地球側のばね定数
                    A_point = Areas[i]                                  #断面積
                    
                #分割区間その１(伸展テザー)　質点2~counter + 10
                elif(N_earth - 12 - counter < i < N_earth - 3):
                    k_m = spring_base* vl.l_0/ vl.l_0_short             #月方向のばね定数
                    k_p = spring_base* vl.l_0/ vl.l_0_short             #地球方向のばね定数
                    A_point = Areas_base                                #断面積  
                            
                #分割区間その２(ノード1分割区間) 質点は10個で確定
                elif(N_earth - 22 - counter < i <= N_earth - 12 - counter):
                    k_m = k_node2[i - (N_earth - 22 - counter) - 1]     #月方向のばね定数
                    k_p = k_node2[i - (N_earth - 22 - counter)]         #地球方向のばね定数
                    A_point = A_divnodes[i - (N_earth - 22 - counter)]  #断面積

                #第一質点(操作質点) ここは完全に伸展テザー
                elif(i == N_earth - 3):
                    code1 = 1
                    l_0 = vl.l_0_short + dl                             #自然長(伸びも考える)
                    #l_0 = max(l_0, vl.l_0_min)                          #最短値を設定(張力の発散を抑制)
                    k_m = spring_base* vl.l_0/vl.l_0_short              #月方向のばね定数
                    k_p = min(spring_base* vl.l_0/l_0,spring_base* vl.l_0/vl.l_0_short*vl.k_max_rate)                       #地球方向のばね定数
                    l_0 = vl.l_0_short
                    A_point = Areas_base                              

                #通常部との境目
                elif(i == N_earth - 22 - counter):
                    code1  = 2
                    l_0 = vl.l_0
                    k_m = spring[-4]                                    #月方向のばね定数
                    k_p = k_node2[0]                                    #地球方向のばね定数
                    A_point = A_divnodes[0]                             #断面積       

            #基礎ノードの分割部が排出され切った点
            elif(counter_2 == 2):
                #質点1を10分割したときのばね定数*10を計算
                A_divnodes = A_node2
                #通常質点
                if(i < N_earth - 22 - counter):
                    l_0 = vl.l_0                                        #自然長(100km基準)
                    k_m = spring[i-1]                                   #月側のばね定数
                    k_p = spring[i]                                     #地球側のばね定数
                    A_point = Areas[i]                                  #断面積
                    
                #分割区間その１(伸展テザー)　質点2~counter + 10
                elif(N_earth - 22 - counter < i < N_earth - 3):
                    k_m = spring_base* vl.l_0/ vl.l_0_short             #月方向のばね定数
                    k_p = spring_base* vl.l_0/ vl.l_0_short             #地球方向のばね定数
                    A_point = Areas_base                                #断面積  

                #第一質点(操作質点) ここは完全に伸展テザー
                elif(i == N_earth - 3):
                    code1 = 1
                    l_0 = vl.l_0_short + dl                             #自然長(伸びも考える)
                    #l_0 = max(l_0, vl.l_0_min)                          #最短値を設定(張力の発散を抑制)
                    k_m = spring_base* vl.l_0/vl.l_0_short              #月方向のばね定数
                    k_p = min(spring_base* vl.l_0/l_0,spring_base* vl.l_0/vl.l_0_short*vl.k_max_rate)                       #地球方向のばね定数
                    l_0 = vl.l_0_short
                    A_point = Areas_base                              

                #通常部との境目
                elif(i == N_earth - 22 - counter):
                    code1  = 2
                    l_0 = vl.l_0
                    k_m = spring[-3]                                    #月方向のばね定数
                    k_p = k_node2[0]                                    #地球方向のばね定数
                    A_point = A_divnodes[0]                             #断面積
        
        else:
            l_0 = vl.l_0
            k_m = spring[i-1]
            k_p = spring[i]
            A_point = Areas[i]
        
        m_point = max(vl.rho * A_point * l_0 * 10**3,10.0)
        damp = 2*(k_p*vl.rho * A_point * l_0 * 10**3)**(1/2)*vl.cr


        r = (vectors[i,0]**2 + vectors[i,1]**2 + vectors[i,2]**2)**(1/2) #月中心からの距離
        r_E = ((vectors[N_earth-1,0] - vectors[i,0])**2 + (vectors[N_earth-1,1] - vectors[i,1])**2 + (vectors[N_earth-1,2] - vectors[i,2])**2)**(1/2) #地球中心からの距離
        rM = ((vectors[i,0]-vectors[i-1,0])**2 + (vectors[i,1]-vectors[i-1,1])**2+ (vectors[i,2]-vectors[i-1,2])**2)**(1/2) #月方向ノードとの距離
        rE = ((vectors[i+1,0]-vectors[i,0])**2 + (vectors[i+1,1]-vectors[i,1])**2+ (vectors[i+1,2]-vectors[i,2])**2)**(1/2) #地球方向ノードとの距離

        Fge = vl.myu/r_E**3 #地球の万有引力
        Fgl = -vl.myu_L/r**3 #月の万有引力

        Fku = k_p/rE*max((rE-l_0),0.0) #地球方向のばね力
        Fkd = k_m/rM*max((rM-l_0),0.0) #月方向のばね力
            
        if(code1 == 1 and case == 0): #月側操作ノード
            Fku = k_p/rE*max((rE-vl.l_0_short),0.0)
        elif(code1 == 2 and case == 0): #地球側境界ノード
            Fku = k_p/rE*max((rE-vl.l_0),0.0)
        elif(code1 == 1 and case == 1):#CW側操作ノード
            Fku = k_p/rE*max((rE-(vl.l_0_short+dl)),0.0)
        elif(code1 == 2 and case == 1): #月側境界ノード
            Fku = k_p/rE*max((rE-vl.l_0_short),0.0)

        Fclimber_u = 0.0
        Fclimber_v = 0.0
        Fclimber_w = 0.0
        #クライマの計算
        #for clim in range(len(climbers)):
            #if(climbers[clim,1]+1 == i):#下方にクライマ
                #距離の計算
                #climvector = (1.0-climbers[clim,2])*vectors[i-1] + climbers[clim,2]*vectors[i]
                #clim_r = (climvector[0]**2 + climvector[1]**2)**(1/2)
                #clim_r_m = ((vectors[vl.N,0]-climvector[0])**2 + (vectors[vl.N,1]-climvector[1])**2)**(1/2)
                #速度の計算
                #clim_u = climbers[clim,0] * (vectors[i,0]-vectors[i-1,0])/rM
                #clim_v = climbers[clim,0] * (vectors[i,1]-vectors[i-1,1])/rM

                #Fclimber_u = vl.myu/clim_r_m**3*(vectors[vl.N,0]-climvector[0]) + (-vl.myu_L/clim_r**3 + omega**2)*climvector[0] + 2*omega*clim_v + alpha*climvector[1] - climbers[clim,4]*climbers[clim,3]*(vectors[i,0]-vectors[i-1,0])/rM + Firr[0]
                #Fclimber_v = vl.myu/clim_r_m**3*(vectors[vl.N,1]-climvector[1]) + (-vl.myu_L/clim_r**3 + omega**2)*climvector[1] - 2*omega*clim_u - alpha*climvector[0] - climbers[clim,4]*climbers[clim,3]*(vectors[i,1]-vectors[i-1,1])/rM + Firr[1]

                #ノードにかかる力更新
                #Fclimber_u = Fclimber_u*vl.Mass_climber/mass[i-1]*climbers[clim,2]
                #Fclimber_v = Fclimber_v*vl.Mass_climber/mass[i-1]*climbers[clim,2]
            
            #elif(climbers[clim,1] == i):#上方にクライマ
                #距離の計算
                #climvector = (1.0-climbers[clim,2])*vectors[i] + climbers[clim,2]*vectors[i+1]
                #clim_r = (climvector[0]**2 + climvector[1]**2)**(1/2)
                #clim_r_m = ((vectors[vl.N,0]-climvector[0])**2 + (vectors[vl.N,1]-climvector[1])**2)**(1/2)
                #速度の計算
                #clim_u = climbers[clim,0] * (vectors[i+1,0]-vectors[i,0])/rE
                #clim_v = climbers[clim,0] * (vectors[i+1,1]-vectors[i,1])/rE

                #Fclimber_u = vl.myu/clim_r_m**3*(vectors[vl.N,0]-climvector[0]) + (-vl.myu_L/clim_r**3 + omega**2)*climvector[0] + 2*omega*clim_v + alpha*climvector[1] - climbers[clim,4]*climbers[clim,3]*(vectors[i+1,0]-vectors[i,0])/rE + Firr[0]
                #Fclimber_v = vl.myu/clim_r_m**3*(vectors[vl.N,1]-climvector[1]) + (-vl.myu_L/clim_r**3 + omega**2)*climvector[1] - 2*omega*clim_u - alpha*climvector[0] - climbers[clim,4]*climbers[clim,3]*(vectors[i+1,1]-vectors[i,1])/rE + Firr[1]

                #ノードにかかる力更新
                #Fclimber_u = Fclimber_u*vl.Mass_climber/mass[i-1]*(1.0-climbers[clim,2])
                #Fclimber_v = Fclimber_v*vl.Mass_climber/mass[i-1]*(1.0-climbers[clim,2])

        ret_Forces[i] = [Fge*m_point*(vectors[N_earth-1,0]-vectors[i,0]),Fge*m_point*(vectors[N_earth-1,1]-vectors[i,1]),Fge*m_point*(vectors[N_earth-1,2]-vectors[i,2]),
                         Fgl*m_point*vectors[i,0],Fgl*m_point*vectors[i,1],Fgl*m_point*vectors[i,2],
                         m_point*vl.omega**2*vectors[i,0],m_point*vl.omega**2*vectors[i,1],0,
                         2*m_point*vl.omega*vectors[i,4],-2*m_point*vl.omega*vectors[i,3],0,
                         Fku*(vectors[i+1,0]-vectors[i,0])/A_point,Fku*(vectors[i+1,1]-vectors[i,1])/A_point,Fku*(vectors[i+1,2]-vectors[i,2])/A_point,
                         Fkd*(vectors[i-1,0]-vectors[i,0])/A_point,Fkd*(vectors[i-1,1]-vectors[i,1])/A_point,Fkd*(vectors[i-1,2]-vectors[i,2])/A_point,
                         damp*(vectors[i+1,3]-vectors[i,3]),damp*(vectors[i+1,4]-vectors[i,4]),damp*(vectors[i+1,5]-vectors[i,5]),
                         damp*(vectors[i-1,3]-vectors[i,3]),damp*(vectors[i-1,4]-vectors[i,4]),damp*(vectors[i-1,5]-vectors[i,5]),
                         Firr[0],Firr[1],Firr[2],
                         Fclimber_u,Fclimber_v,Fclimber_w
                        ]
    #CW
    #月面操作
    r = (vectors[N_earth-2,0]**2 + vectors[N_earth-2,1]**2 + vectors[N_earth-2,2]**2)**(3/2) #月からの距離
    r_E = ((vectors[N_earth-1,0] - vectors[N_earth-2,0])**2 + (vectors[N_earth-1,1] - vectors[N_earth-2,1])**2 + (vectors[N_earth-1,2] - vectors[N_earth-2,2])**2)**(3/2)
    rM = ((vectors[N_earth-2,0]-vectors[N_earth-3,0])**2 + (vectors[N_earth-2,1]-vectors[N_earth-3,1])**2 + (vectors[N_earth-2,2]-vectors[N_earth-3,2])**2)**(1/2)

    Fge = vl.myu/r_E
    Fgl = -vl.myu_L/r
    if(case == 0):
        Fkd = spring[-3]/rM*max((rM-vl.l_0),0.0)
    elif(case == 1):
        l_0 = vl.l_0_short + dl                     #自然長(伸びも考える)
        #l_0 = max(l_0,vl.l_0_min)                   #最短値を設定(張力の発散を抑制)
        k_p = min(spring[-1]*vl.l_0/l_0,spring[-1]*vl.l_0/vl.l_0_short*vl.k_max_rate)                 #地球方向のばね定数
        Fkd = k_p/rM*max((rM-l_0),0.0)

    Fclimber_u = 0.0
    Fclimber_v = 0.0
    Fclimber_w = 0.0

    #クライマの計算
    #for clim in range(len(climbers)):
        #if(climbers[clim,1]+1 == vl.N-2):#下方にクライマ
            #距離の計算
            #climvector = (1.0-climbers[clim,2])*vectors[vl.N-2] + climbers[clim,2]*vectors[vl.N]
            #clim_r = (climvector[0]**2 + climvector[1]**2)**(1/2)
            #clim_r_m = ((vectors[vl.N,0]-climvector[0])**2 + (vectors[vl.N,0]-climvector[1])**2)**(1/2)
            #速度の計算
            #clim_u = climbers[clim,0] * (vectors[vl.N-1,0]-vectors[vl.N-2,0])/rM
            #clim_v = climbers[clim,0] * (vectors[vl.N-1,1]-vectors[vl.N-2,1])/rM

            #Fclimber_u = vl.myu/clim_r_m**3*(vectors[vl.N,0]-climvector[0]) + (-vl.myu_L/clim_r**3 + omega**2)*climvector[0] + 2*omega*clim_v + alpha*climvector[1] - climbers[clim,4]*climbers[clim,3]*(vectors[vl.N-1,0]-vectors[vl.N-2,0])/rM + Firr[0]
            #Fclimber_v = vl.myu/clim_r_m**3*(vectors[vl.N,1]-climvector[1]) + (-vl.myu_L/clim_r**3 + omega**2)*climvector[1] - 2*omega*clim_u - alpha*climvector[0] - climbers[clim,4]*climbers[clim,3]*(vectors[vl.N-1,1]-vectors[vl.N-2,1])/rM + Firr[1]

            #ノードにかかる力更新
            #Fclimber_u = Fclimber_u*vl.Mass_climber/(Mcw+mass[vl.N-1])*climbers[clim,2]
            #Fclimber_v = Fclimber_v*vl.Mass_climber/(Mcw+mass[vl.N-1])*climbers[clim,2]

    ret_Forces[N_earth-2] = [Fge*(vl.Mcw+mass[-1])*(vectors[N_earth-1,0]-vectors[N_earth-2,0]),Fge*(vl.Mcw+mass[-1])*(vectors[N_earth-1,1]-vectors[N_earth-2,1]),Fge*(vl.Mcw+mass[-1])*(vectors[N_earth-1,2]-vectors[N_earth-2,2]),
                       Fgl*(vl.Mcw+mass[-1])*vectors[N_earth-2,0],Fgl*(vl.Mcw+mass[-1])*vectors[N_earth-2,1],Fgl*(vl.Mcw+mass[-1])*vectors[N_earth-2,2],
                       (vl.Mcw+mass[-1])*vl.omega**2*vectors[N_earth-2,0],(vl.Mcw+mass[-1])*vl.omega**2*vectors[N_earth-2,1],0,
                       2*(vl.Mcw+mass[-1])*vl.omega*vectors[N_earth-2,4],-2*(vl.Mcw+mass[-1])*vl.omega*vectors[N_earth-2,3],0,
                       0.0,0.0,0.0,
                       Fkd*(vectors[N_earth-3,0]-vectors[N_earth-2,0])/Areas[-1],Fkd*(vectors[N_earth-3,1]-vectors[N_earth-2,1])/Areas[-1],Fkd*(vectors[N_earth-3,2]-vectors[N_earth-2,2])/Areas[-1],
                       0.0,0.0,0.0,
                       damps[-2]*(vectors[N_earth-3,3]-vectors[N_earth-2,3]),damps[-2]*(vectors[N_earth-3,4]-vectors[N_earth-2,4]),damps[-2]*(vectors[N_earth-3,5]-vectors[N_earth-2,5]),
                       Firr[0],Firr[1],Firr[2],
                       Fclimber_u,Fclimber_v,Fclimber_w
                       ]
    
    #Earth
    r = (vectors[N_earth-1,0]**2 + vectors[N_earth-1,1]**2 + vectors[N_earth-1,2]**2)**(3/2)
    ret_Forces[N_earth-1] = [
        0.0,0.0,0.0,
        vl.myu_L/r*vectors[N_earth-1,0],vl.myu_L/r*vectors[N_earth-1,1],vl.myu_L/r*vectors[N_earth-1,2],
        vl.omega**2*vectors[N_earth-1,0],vl.omega**2*vectors[N_earth-1,1],0,
        2*vl.omega*vectors[N_earth-1,4],-2*vl.omega*vectors[N_earth-1,3],0,
        0.0,0.0,0.0,
        0.0,0.0,0.0,
        0.0,0.0,0.0,
        0.0,0.0,0.0,
        Firr[0],Firr[1],Firr[2],
        0.0,0.0,0.0
    ]

    return ret_Forces