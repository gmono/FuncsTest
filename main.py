from tools import *
from sklearn import neural_network
from nn import FLReg,NNReg,ZHReg
reg=ZHReg(5)
from funcs import *
# reg=neural_network.MLPRegressor(hidden_layer_sizes=(5,10,25,50,100,250,500,1000),activation="relu",solver="lbfgs")
contrast_reg(sanjiao,reg,start=0,end=50,rstart=0,rend=200)
for i in reg.collect_params():
    print(i)
    print(reg.collect_params()[i].data())