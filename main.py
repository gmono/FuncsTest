from tools import *
from sklearn import svm,tree
from nn import *
reg=FLReg(5)
from funcs import *
#通用测试
# reg=svm.SVR(gamma=0.00011,C=10)
# reg=tree.DecisionTreeRegressor(max_depth=5)
# contrast_reg(fang,reg,start=-10,end=10,rstart=-20,rend=20)
# for i in reg.collect_params():
#     print(i)
#     print(reg.collect_params()[i].data())

#神经网络测试函数
xs,ys,_=getdata(fang,0,10)
xs=xs.astype("float32")
ys=ys.astype("float32")
fit(reg,xs,ys,draw_pars=[0,10,-20,20,fang])
import  os
os.system("pause")