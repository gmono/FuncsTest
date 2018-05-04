from tools import *
from sklearn import svm,tree
from nn import *
reg=NNReg()
from funcs import *
#通用测试
# reg=svm.SVR(gamma=0.00011,C=10)
# reg=tree.DecisionTreeRegressor(max_depth=5)
# contrast_reg(fang,reg,start=-10,end=10,rstart=-20,rend=20)
# for i in reg.collect_params():
#     print(i)
#     print(reg.collect_params()[i].data())

#神经网络测试函数
data,zs=get2DData(sq_2d,0,10,0,10)
data=data.astype("float32")
zs=zs.astype("float32")
fit(reg,data,zs,draw_pars=[0,5,0,8,sq_2d],drfunc=plot_3d)
import  os
os.system("pause")
