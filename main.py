from tools import *
from sklearn import svm,tree
from nn import *
reg=FLReg()
from funcs import *
# reg=svm.SVR(gamma=0.00011,C=10)
# reg=tree.DecisionTreeRegressor(max_depth=5)
contrast_reg(sanjiao,reg,start=-10,end=10,rstart=-20,rend=20)
# for i in reg.collect_params():
#     print(i)
#     print(reg.collect_params()[i].data())

import  os
os.system("pause")