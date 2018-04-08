from mxnet import gluon
nn=gluon.nn
from numpy import *
from mxnet.gluon import Trainer
from mxnet.gluon import data as mxdata
from mxnet import autograd as ag
from mxnet.gluon import loss
from mxnet import nd
from mxnet import initializer
def fit(nn:nn.Block,xs,ys,batchsize=10):
    ds=mxdata.ArrayDataset(xs,ys)
    dl=mxdata.DataLoader(ds,batch_size=batchsize,shuffle=True)
    tr=Trainer(nn.collect_params(),optimizer="rmsprop",optimizer_params={"learning_rate":0.1})
    lfunc=loss.L1Loss()
    for i in range(200):
        for data,label in dl:
            with ag.record():
                y=nn(data)
                ls=lfunc(y,label) #type:nd.NDArray
            ls.backward()
            tr.step(batch_size=batchsize,ignore_stale_grad=True)

class FunReg(nn.Block):

    def fit(self,xs,ys):
        xs=xs.astype("float32")
        ys=ys.astype("float32")
        ys=transpose(ys)
        fit(self,xs,ys,batchsize=20)

    def predict(self,xs):
        xs = xs.astype("float32")
        ys=self(nd.array(xs)).asnumpy()
        return transpose(ys)[0]


class FLReg(FunReg):
    """傅里叶级数拟合器"""
    def __init__(self,n=10):
        super(FLReg, self).__init__()
        #自有参数
        self.n=n
        self.l=self.params.get("l",shape=(1,))
        self.an=self.params.get("an",shape=(n,))
        self.bn = self.params.get("bn", shape=(n,))
        self.initialize(init="ones")
    def forward(self, x,*args):
        #傅里叶级数
        x=nd.array(x)
        ret=self.an.data()[0] #type:nd.NDArray
        ret=ret.broadcast_to(shape=x.shape)
        for i in range(1,self.n):
            adata=self.an.data()
            bdata=self.bn.data()
            ret=adata[i]*nd.cos((i*pi*x)/self.l.data())+ret
            ret=bdata[i]*nd.sin((i*pi*x)/self.l.data())+ret
        return ret


class NNReg(FunReg):
    """神经网络拟合器"""
    def __init__(self):
        super(NNReg,self).__init__()
        self.nn=nn.Sequential()
        with self.nn.name_scope():
            self.nn.add(nn.Dense(100,activation="relu"))
            self.nn.add(nn.Dense(100,activation="relu"))
            self.nn.add(nn.Dense(1))
        self.register_child(self.nn)
        self.initialize(init=initializer.Zero())

    def forward(self, x,*args):
        return self.nn(x)

class ZHReg(FunReg):
    """综合拟合器"""
    def __init__(self,fln=10):
        super(ZHReg, self).__init__()
        self.fl=FLReg(fln)
        self.nn=NNReg()
        self.nn2=NNReg()
        self.register_child(self.fl)
        self.register_child(self.nn)
    def forward(self, x,*args):
        flret=self.fl(x)
        nnret=self.nn(x)
        nn2ret=self.nn2(x)
        return flret*nnret+nn2ret
