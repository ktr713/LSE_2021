from LSEmodules import calcForce
from LSEmodules import calctension
from LSEmodules import rk4
from LSEmodules import values as vl
import time
import os
import numpy as np
from numba import jit, f8, i8, b1, void
from plyer import notification

#--------------------------------------------------------------------------------------------------------------------------------
#操作変数のまとめ
#月面操作(0) or CW操作(1) or 加工無し(else)
case_extension = 0

#速度調整法:段階調整法(0) or 相対速度制御(1) or 張力制御(2) or 展開無し(3)
case_velocity = 1

#メイン開始
#入出力処理
#実行時間の決定
#ステップ間隔 基本は0.1秒
dt = 0.1
days = 28
tmax = vl.timer(days,dt)
cuttime1 = int(3600*0.5*1/dt)

#保存フォルダの作成　フォルダ名はlog¥¥(year,month,day,hour,min,secの最大14桁)+(素材名)+(実行日数) + day¥¥
T = time.localtime()
times = ""

for k in range(6):
    times += str(T[k])
    
times = "log\\" + times + vl.material + str(days) + "days"
os.mkdir(times)
os.mkdir(times + "\\system")
os.mkdir(times + "\\forces")
os.mkdir(times + "\\climbers")

#初期化
init_vectors = np.zeros((vl.N+1,6))

k_ratio = vl.k_ratios
masses = vl.masses
damps = vl.damps
Areas = vl.Areas
    
#クライマのデータを格納する行列
climber = np.array([[0.0,0.0,0.0,0.0,1.0,0.0]]) #(Velocity,nodenumber,long_ratio,accel,up/down,starttime)
#クライマ座標出力行列
climber_output = np.array([[[0,0,0,0,0],[0,0,0,0,0]],[[0,0,0,0,0],]])

#初期値入力 自然長から始める時は上側，既知のデータを用いる時は下から．片方はコメントアウトしておくこと
#for i in range(vl.N+1):
    #init_vectors[i,1] = 1737.0 + vl.L/float(vl.N)*i
init_vectors = np.loadtxt("inputdata\\zylon\\perigee\\initdata_3dim.csv",delimiter=",")
#init_vectors[:,4] += 1  #1なら通常質点
#init_vectors[vl.N-1,4] = 0 #0ならCW

#地球の初期座標　(1-0.0527)で近地点，(1+0.0527)で遠地点   初期速度 近地点/遠地点ならvy=0,vx = r×(Ω-ω)
init_R = vl.r_L*(1-0.0527)
init_vectors[vl.N,1] = init_R
init_vectors[vl.N,3] = init_R*(vl.omega-vl.alpha/init_R**2)

#座標加工(質点の分割)
vectors = vl.divide_vectors(init_vectors,case_extension)

#加工完了
#出力処理
#座標行列をnumpy.array型に変換,出力
np.savetxt(times + "\\initdata.csv",vectors,delimiter=",")

#出力ファイルの指定,作成
files = open(times +"\\nodes.csv","w")
files.write("basex,basey,basez,500x,500y,500z,CWx,CWy,CWz,earthx,earthy,earthz")
files.write("\n")

files2 =  open(times +"\\v_rate.csv","w")
files2.write("根本速度,dl,counter,counter_2")
files2.write("\n")

files3 = open(times + "\\energy.csv","w")
files3.write("時間,根本張力,動力")
files3.write("\n")

#設定項目，備考等をtxt形式で出力 string.write()のなかに適当に記述
setting = open(times + "\\setting.txt","w")
setting.write("3次元地球軌道計算_テスト")
setting.close()

print("実行中test" + __file__)

#繰り出し/削除質点のカウンタ
counter = 0
#追加/削除質点のカウンタ
counter_2 = 0
#操作長
dl = 0.0
#総エネルギー
en_all = 0.0
#正の動力
en_all_p = 0.0
#負の動力
en_all_m = 0.0
#一時保存速度
v_tmp = 0.0
#解析開始時間
startdays = 0

#v_tether = 100.0/(24*3600*27.3217/2)
v_tether = 0.0
#v_rate = 2000.0/40551.23
v_rate = 1.0
v_tmp = 0.0

#forces = calcForce.Forces(vectors,k_ratio,masses,damps,Areas,climber,counter,counter_2,dl,case_extension)
#np.savetxt(times + "\\forces\\test.csv",forces,delimiter=",")

#-----------------------------------------------------------途中から読み込む場合-------------------------------------------------------------
if(False): #True/Falseを手書きで切り替えること
    vectors = np.loadtxt("inputdata\\zylon\\perigee\\initdata_test2.csv",delimiter=",")
    counter = 3
    counter_2 = 55
    dl = -9.901
    startdays = 14.0
    v_tether = -0.00124968499460505
    v_tmp = 0.00508346438504518

#-------------------------------------------------------------------------------------------------------------------------------------------

#計測開始!
start = time.time()

en = 0.0
en_p = 0.0
en_all = 0.0
en_all_p = 0.0
en_all_m = 0.0

#---------------------------------------------------------------メイン部---------------------------------------------------------------------
#t-loopスタート
for t in range(tmax):
    #rk法適用，dt秒後の座標データ (座標ベクトル,質量行列,バネ系数行列,減衰係数,クライマ行列,ステップ数,基地付近の質点変化量,全体の質点変化量,1km以下の伸縮)
    vectors = rk4.rk4(vectors,masses,k_ratio,Areas,damps,climber,dt,counter,counter_2,dl,case_extension)
    if(case_extension == 0):
        en = min(k_ratio[0]*vl.l_0/(vl.l_0_short+dl),k_ratio[0]*vl.l_0/vl.l_0_short*vl.k_max_rate)*max(((vectors[1,0]-vectors[0,0])**2 + (vectors[1,1]-vectors[0,1])**2 + (vectors[1,2]-vectors[0,2])**2)**(1/2)-(vl.l_0_short + dl),0.0)
        if(en > 50):
            en = k_ratio[1]*vl.l_0/vl.l_0_short*max(((vectors[2,0]-vectors[1,0])**2 + (vectors[2,1]-vectors[1,1])**2 + (vectors[2,2]-vectors[1,2])**2)**(1/2)-vl.l_0_short,0.0)
        en_p = en*v_tether
        en_all += en_p
        en_all_p += en_p if np.sign(en_p) == 1 else 0.0
        en_all_m += en_p if np.sign(en_p) == -1 else 0.0

    #cuttime1(デフォルトで1時間)ごとに座標，力のデータを出力 クライマはコメントアウト   
    if t%cuttime1 == 0: 
        forces = calcForce.Forces(vectors,k_ratio,masses,damps,Areas,climber,counter,counter_2,dl,case_extension)
        np.savetxt(times +"\\system\\{}daysdata.csv".format(t/(3600*24*1/dt)),vectors,delimiter=",")
        np.savetxt(times + "\\forces\\{}daysforces.csv".format(t/(3600*24*1/dt)),forces,delimiter=",")
        
        #n=1,500,CW,Earthの座標データ出力
        Base_str = [str(n) for n in vectors[1]]
        Nodex_str = [str(n) for n in vectors[500]]
        Nodex2_str = [str(n) for n in vectors[-2]]
        CW_str = [str(n) for n in vectors[-1]]

        writedatas = [Base_str[0],Base_str[1],Base_str[2],Nodex_str[0],Nodex_str[1],Nodex_str[2],Nodex2_str[0],Nodex2_str[1],Nodex2_str[2],CW_str[0],CW_str[1],CW_str[2]]

        files.write(','.join(writedatas))
        files.write("\n")
        files2.write(str(v_tether) + "," + str(dl) + "," + str(counter) + "," + str(counter_2))
        files2.write("\n")
        files3.write(str(t/(3600*24*1/dt)) + "," + str(en) + "," + str(en_all) + "," + str(en_all_p) + "," + str(en_all_m))
        files3.write("\n")
    
    checker = 1
    if(t>vl.t_return - startdays*24*3600*1/dt):checker = -1
    #----------------------------------------------------------伸縮速度調整------------------------------------------------------------------
    #いくつか調整パターンを用意 一つ使うときはその他をコメントアウトしておくこと
    
    #段階調整(規定時間で速度を切り替える)
    if(case_velocity == 0):
        v_tether,v_tmp = vl.velocity_pulse(vectors,t,dt,v_tether,v_tmp,startdays)
    #相対速度調整(地球の相対速度を参考に速度調整)
    elif(case_velocity == 1):
        v_tether = vl.velocity_earth(vectors,t,dt,v_tether)
    #張力制御
    elif(case_velocity == 2):
        v_tether,v_tmp = vl.velocity_tension(vectors,t,dt,v_tether,v_tmp,dl,startdays)
    #調整なし(test)
    elif(case_velocity == 3):
        v_tether = 0.0
    
    #----------------------------------------------------------伸縮処理------------------------------------------------------------------
    if(case_extension == 0):
        vectors,counters = vl.change_nodes_moon(vectors,t,dt,[v_tether,dl,counter,counter_2],startdays)
        dl = counters[0]
        counter = counters[1]
        counter_2 = counters[2]

    elif(case_extension == 1):
        vectors,counters = vl.change_nodes_cw(vectors,t,dt,[v_tether,dl,counter,counter_2],startdays)
        dl = counters[0]
        counter = counters[1]
        counter_2 = counters[2]

    #-------------------------------------------------------------------------------------------------------------------------------------
    if(t%1000 == 0):
        print("printed {} days data\n".format(t/(1/dt*3600.0*24.0)))
        print("sigma's ret is {},counter is {}".format(checker,counter))
    '''
    if((vectors[1,0]**2+vectors[1,1]**2)**(1/2) < 1737.0):
        print("Brake!")
        break
    '''

#クライマ座標データの出力
#np.savetxt(times+"\\climvector1.csv",climber_output[0],delimiter=",")
#np.savetxt(times+"\\climvector2.csv",climber_output[1],delimiter=",")

files.close()
files2.close()
files3.close()
    
fin = (time.time() - start)/3600.0
notification.notify(
    title ="program complete!",
    message = "time = {}".format(fin),
    app_name = os.path.basename(__file__),
    app_icon = "inputdata\\zylon\\perigee\\icon.ico"
)
print(fin)