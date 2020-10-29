import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

x0 = (0, 1)
x1 = (1, 0)
r = 0.5
l = (np.sqrt(2) / r) - 1 

theta = np.linspace(0,2*3.1415926, 100)
u = x1[0] + r * np.cos(theta)
v = x1[1] + r * np.sin(theta)


def fun0(x, y):
    return (x - x0[0]) ** 2 + (y - x0[1]) ** 2


def fun1(x, y):
    return fun0(x, y) + l * ((x - x1[0]) ** 2 + (y - x1[1]) ** 2 - r ** 2)


def fun2(x, y):
    return fun0(x, y) + 0.1 * ((x - x1[0]) ** 2 + (y - x1[1]) ** 2 - r ** 2)


# ======== 等高線 ========
n = 256
x = y = np.linspace( -3, 3, n)
# 將原始資料變成網格資料
X,Y = np.meshgrid(x,y)
zs = np.array([fun0(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
ws = np.array([fun2(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
W = ws.reshape(X.shape)
# 填充顏色
plt.figure(figsize=(10,10))
# plt.contourf(X,Y,Z,100,alpha=0.5)

# add contour lines
C = plt.contour(X,Y,Z,100, colors='b')
D = plt.contour(X,Y,W,100, colors='r')

plt.scatter(x0[0], x0[1], color='b')
plt.scatter(x1[0], x1[1], color='g')
plt.scatter(u, v, color='g', s=5)

# 顯示各等高線的資料標籤cmap=plt.cm.hot
plt.clabel(C, inline=True,fontsize=10)
plt.clabel(D, inline=True,fontsize=10)

plt.show()


# # =============== 3D plot ===============
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x = y = np.arange(-3.0, 3.0, 0.05)
# X, Y = np.meshgrid(x, y)
# zs = np.array([fun2(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
# Z = zs.reshape(X.shape)
# ax.plot_surface(X, Y, Z, color='b')

# ws = np.array([fun1(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
# W = ws.reshape(X.shape)
# ax.plot_surface(X, Y, W, color='r')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.legend()
# plt.show()


# =============== 2D plot with lambad at xm, ym ===============
def fun(x, y, ld):
    return fun0(x, y) + ld * ((x - x1[0]) ** 2 + (y - x1[1]) ** 2 - r ** 2)

x0 = (0, 1)
x1 = (1, 0)
r = 0.5
lm = (np.sqrt(2) / r) - 1

xm = 1 - r/np.sqrt(2)
ym = 0 + r/np.sqrt(2)
ls = np.arange(-3.0, 3.0, 0.05)
zs = [fun(xm, ym, w) for w in ls]
# plt.scatter(ls, zs)
# plt.show()


# =============== 2D plot with G(ld) = inf_{x,y} fun(x, y, ld) ===============
ld_list = np.arange(0, 10.0, 0.05)
r_list = [np.sqrt(2) / (l+1) for l in ld_list]
x_list = [1 - a/np.sqrt(2) for a in r_list]
y_list = [0 + a/np.sqrt(2) for a in r_list]
f_list = [fun(x, y, ld) for x, y, ld in zip(x_list, y_list, ld_list)]
plt.scatter(ld_list, f_list)
plt.scatter(lm, fun(xm, ym, lm), color='r')
plt.show()
