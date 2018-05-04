
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
# from mayavi import mlab
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

def get2DData(func,xstart,xend,ystart,yend):
    """
    获取2元函数的data和z值

    :param func: 二元函数
    :param xstart: x起始点
    :param xend: x结束点
    :param ystart: y起始点
    :param yend: y结束点
    :return: data,z
    """
    xs=np.linspace(xstart,xend,(xend-xstart))
    ys=np.linspace(ystart,yend,(yend-ystart))
    data=list(itertools.product(xs,ys))
    xy=np.array(list(zip(*data)))
    zs=func(xy[0],xy[1])
    return np.array(data),zs

def get2DRegData(reg,xstart,xend,ystart,yend):
    """和上面的区别：这里使用product后的一个二维点列表来放入回归器中处理，而不是直接提供x列表和y列表"""
    xs=np.linspace(xstart,xend,(xend-xstart))
    ys=np.linspace(ystart,yend,(yend-ystart))
    data=np.array(list(itertools.product(xs,ys)))
    zs=reg.predict(data)
    return data,zs

#统一绘图函数
def plot_2d(func,reg,start=-100,end=100,rstart=-100,rend=100):
    #训练数据和训练
    xs,ys,xsf=getdata(func,start,end)
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

def plot_3d(func,reg,start=-100,end=100,rstart=-100,rend=100):
    """
    3d绘图函数
    :param func: 原始函数
    :param reg: 回归器
    :param start: x和y的开始点
    :param end: x和y的结束点
    :param rstart: 增大范围测试的xy开始点
    :param rend: 相应结束点
    :return: 无
    """
    tplt=plt.gcf()
    #下面两个一个是start和end 一个是rstart和rend
    s1=tplt.add_subplot(121,projection="3d")
    s2=tplt.add_subplot(122,projection="3d")
    def plot(p,tstart,tend):
        _,z1_1=get2DData(func,tstart,tend,tstart,tend)
        p.plot_surface(*np.mgrid[tstart:tend,tstart:tend],z1_1.reshape((tend-tstart,tend-tstart)))
        _,z1_2=get2DRegData(reg,tstart,tend,tstart,tend)
        p.plot_surface(*np.mgrid[tstart:tend, tstart:tend], z1_2.reshape((tend - tstart, tend - tstart)))
    plot(s1,start,end)
    plot(s2,rstart,rend)


#统一比较函数

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

def contrast_3d(func,reg,start=-100,end=100,rstart=-100,rend=100):
    #训练
    data,zs=get2DData(func,start,end,start,end)
    reg.fit(data,zs.reshape((-1,1)))
    #绘图
    plot_3d(func,reg,start,end,rstart,rend)
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
