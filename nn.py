from mxnet import gluon
nn=gluon.nn
from numpy import *
from mxnet.gluon import Trainer
from mxnet.gluon import data as mxdata
from mxnet import autograd as ag
from mxnet.gluon import loss
from mxnet import nd
from mxnet import initializer
from matplotlib import pyplot as plt
from tools import *
from funcs import *

def fit(nn:nn.Block,xs,ys,batchsize=20,draw_pars=None,drfunc=None):
    """训练函数"""
    ds=mxdata.ArrayDataset(xs,ys)
    dl=mxdata.DataLoader(ds,batch_size=batchsize,shuffle=True)

    ###
    tr=Trainer(nn.collect_params(),optimizer="rmsprop",optimizer_params={"learning_rate":0.001})
    ###
    lfunc=loss.L2Loss()
    for i in range(2000):
        for data,label in dl:
            with ag.record():
                y=nn(data)
                ls=lfunc(y,label).mean() #type:nd.NDArray
            ls.backward()
            tr.step(batch_size=batchsize,ignore_stale_grad=True)
        print(f"Loss值:{ls.asscalar()}")
        # if ls.asscalar()<0.1:
        #     break
        # 绘图

        if draw_pars is not None and drfunc is not None and ls.asscalar()<10:
            plt.ion()
            plt.gcf().clear()
            drfunc(draw_pars[4], nn, draw_pars[0], draw_pars[1], draw_pars[2], draw_pars[3])
            plt.pause(0.5)



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
        #L只能是正数
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
            ret=adata[i]*nd.cos((i*pi*x)/self.l.data().abs())+ret
            ret=bdata[i]*nd.sin((i*pi*x)/self.l.data().abs())+ret
        return ret

class FastFLReg(FunReg):
    """
    快速版傅里叶级数拟合器
    使用向量化技术
    """
    def __init__(self,n=10):
        super(FastFLReg, self).__init__()
        #自有参数
        self.n=n
        #L只能是正数
        self.l=self.params.get("l",shape=(1,))
        self.an=self.params.get("an",shape=(n,))
        self.bn = self.params.get("bn", shape=(n,))
        self.initialize(init="ones")
    def forward(self, x,*args):
        #傅里叶级数
        x=nd.array(x)
        n=self.n
        ns=nd.array(range(1,n))
        ns=ns.reshape((-1,1))
        T=nd.dot(ns,x.reshape((1,-1)))
        pl=2*pi/self.l.data().abs()
        #
        an=self.an.data()
        bn=self.bn.data()
        san=an[1:].reshape((-1,1))
        sbn=bn[1:].reshape((-1,1))
        f=san*nd.cos(T*pl)+sbn*nd.sin(T*pl)
        f=nd.sum(f,axis=0,keepdims=False)
        f=f+an[0]
        return f.reshape((-1,1))


class CasCadeFastFLReg(FunReg):
    """
    级联增长 先训练前面的
    快速傅里叶级数拟合器
    使用向量化技术
    """
    def __init__(self,n=10,add_count=100):
        """
        :param n: 层数
        :param add_count: 训练多少个batch 后增加一层 一直到10层
        """
        super(CasCadeFastFLReg, self).__init__()
        #cas
        self.add_count=add_count
        self.now_count=0
        self.now_n=1 #1->n之间 则计算时使用0到n-1的元素
        #自有参数
        self.n=n
        #L只能是正数
        self.l=self.params.get("l",shape=(1,))
        self.an=self.params.get("an",shape=(n,))
        self.bn = self.params.get("bn", shape=(n,))
        self.initialize(init="ones")
    def log(self):
        if self.now_n==self.n-1:
            return
        self.now_count+=1
        if self.now_count==self.add_count:
            self.now_count=0
            self.now_n+=1


    def forward(self, x,*args):
        #傅里叶级数
        x=nd.array(x)
        n=self.n
        ns=nd.array(range(1,n))
        ns=ns.reshape((-1,1))
        T=nd.dot(ns,x.reshape((1,-1)))
        pl=2*pi/self.l.data().abs()
        #
        an=self.an.data()
        bn=self.bn.data()
        san=an[1:].reshape((-1,1))
        sbn=bn[1:].reshape((-1,1))
        f=san*nd.cos(T*pl)+sbn*nd.sin(T*pl)
        #进行Dropout操作 实现Cascade
        f=f[:self.now_n,:]
        ####
        f=nd.sum(f,axis=0,keepdims=False)
        f=f+an[0]
        ###记录
        self.log()
        return f.reshape((-1,1))



class NNReg(FunReg):
    """神经网络拟合器"""
    def __init__(self):
        super(NNReg,self).__init__()
        self.nn=nn.Sequential()
        with self.nn.name_scope():
            self.nn.add(nn.Dense(100))
            self.nn.add(nn.LeakyReLU(0.1))
            self.nn.add(nn.Dropout(0.1))
            self.nn.add(nn.Dense(100))
            self.nn.add(nn.LeakyReLU(0.1))
            self.nn.add(nn.Dropout(0.1))
            self.nn.add(nn.Dense(1))
        self.register_child(self.nn)
        self.initialize()

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
        self.register_child(self.nn2)
    def forward(self, x,*args):
        flret=self.fl(x)
        nnret=self.nn(x)
        nn2ret=self.nn2(x)
        return flret*nnret+nn2ret


class AdditionNNReg(FunReg):
    def __init__(self,n=3):
        super().__init__()
        self.nns=[]
        #添加一个
        for i in range(n):
            with self.name_scope():
                tnn = NNReg()
                self.nns.append(tnn)
                self.register_child(tnn)

    def forward(self, x,*args):
        ret=None
        for nn in self.nns:
            if ret is None:
                ret=nn(x)
            else:
                ret=ret+nn(x)
        return ret

class MultiplyNNReg(FunReg):
    def __init__(self,n=3):
        super().__init__()
        self.nns=[]
        #添加一个
        for i in range(n):
            with self.name_scope():
                tnn = NNReg()
                self.nns.append(tnn)
                self.register_child(tnn)

    def forward(self, x,*args):
        ret=None
        for nn in self.nns:
            if ret is None:
                ret=nn(x)
            else:
                ret=ret*nn(x)
        return ret

