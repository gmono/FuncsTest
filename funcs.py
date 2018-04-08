import numpy as np
fang=lambda x:np.sign(abs(x)%10-5)
sanjiao=lambda x:1.5*np.sin(np.cos(x))*np.sin(x)
duoxiangshi=lambda x:1.5*x**4+2.5*x**3+4.5*x**2+0.5*x