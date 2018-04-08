
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# plt.ion()
# from mayavi import mlab
def getdata(func,start=-100,end=100):
    """
    返回一个测试数据集 为func函数的曲线 （单变量）
    :param func: 函数
    :param start:  开始的x
    :param end:  结束的x
    :return: 训练数据 训练标签 原始数据（列表）
    """
    xs=np.linspace(start,end,(end-start)*10)
    ys=func(xs)
    return xs.reshape((-1,1)),ys,xs

def getRegData(reg,start=-100,end=100):
    xs = np.linspace(start, end, (end - start) * 10)
    ys = reg.predict(xs.reshape(-1,1))
    return xs.reshape((-1, 1)), ys, xs

def contrast_reg(func,reg,start=-100,end=100,rstart=-100,rend=100):
    #训练数据和训练
    xs,ys,xsf=getdata(func,start,end)
    reg.fit(xs,ys)
    #同等比例比较
    rxs,rys,rxsf=getRegData(reg,start,end)
    p1=plt.subplot(121)
    p1.plot(xsf,ys)
    p1.plot(rxsf,rys)
    #延伸比较
    xs,ys,xsf=getdata(func,rstart,rend)
    rxs,rys,rxsf=getRegData(reg,rstart,rend)
    p2=plt.subplot(122)
    p2.plot(xsf,ys)
    p2.plot(rxsf,rys)
    plt.show()

def contrast_2t3d(func,start=-100,end=100):
    xs,ys,xsf=getdata(func,start,end)
    ymax,ymin=np.max(ys),np.min(ys)
    [tx,ty]=np.meshgrid(xs,np.linspace(ymin,ymax,(ymax-ymin)*10))
    #获得稀疏版本用于绘图
    # mtx=np.array([tx[0]])
    # mty=ty[:,0].reshape((ty[:,0].shape[0],1))
    #得到z
    tz=ty-ys
    #2d图
    d2=plt.gcf().add_subplot(111)
    d2.plot(xs,ys)
    #3d图
    # mlab.surf(mtx,mty,tz)
    d3=plt.gcf().add_subplot(121,projection="3d")
    d3.plot_surface(tx,ty,tz)
    plt.show()
