import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


x0 = 0
x1 = 1
r = 0.5

if r <= np.abs(x1 - x0):
    xm = x1 - r
    lm = (1 / r) - 1
    ls = 0
else:
    xm = 0
    lm = 0
    ls = 0.


def f0_fun(x):
    return x ** 4 - 50 * (x ** 2) + 100 * x


def g_fun(x):
    return -x - 2.5


def lagrangian(x, l):
    return f0_fun(x) + l * g_fun(x)



# =============== 2D plot ===============
# x = np.arange(-2.0, 2.0, 0.05)
# f0_list = [f0_fun(t) for t in x ]
# g_list = [g_fun(t) for t in x]
# lgg_list = [lagrangian(t, lm) for t in x]
# plt.scatter(x, f0_list, c='k', s=5, label='f0(x)')
# plt.scatter(x, g_list, c='b', s=5, label='g(x)')
# plt.scatter(x, lgg_list, c='r', s=5, label='lgg(x, lm)')
# plt.scatter(xm, 0, c='g', s=50)
# plt.legend()
# plt.show()



# ================================== Problems ==================================
"""
------------------ Problem 1 ------------------
    min f0(x)
    s.t. g(x) <= 0


This problem is equal to 
    inf_x sup_{l>0} L(x, l), 
where L(x, l) = f0(x) + l * g(x)

proof:
    if x do not satisfy g(x) <= 0, then sup_{l>0} L(x, l) = +inf   .... 1.1
    if x satisfies g(x) <= 0, then sup_{l>0} L(x, l) = f(x)        .... 1.2
    from e.q. 1.1 and 1.2, we have
    arg_x inf_x sup_{l>0} L(x, l) = arg_x Problem 1


------------------ Problem 2 ------------------
min f0(x) + lm * g(x)


------------------ Problem 3 ------------------
Gradient L(x, l) = 0


------------------ Problem 4 ------------------
sup_{l>0} inf_x L(x, l)

"""



# ================================== 3D plots ==================================
x = np.arange(-100., 100., 1)
l = np.arange(ls, 100.0, 1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, L = np.meshgrid(x, l)
zs = np.array([lagrangian(xi, li) for xi, li in zip(np.ravel(X), np.ravel(L))])
Z = zs.reshape(X.shape)
ax.plot_surface(X, L, Z, color='lightgray')

ax.scatter(xm, lm, lagrangian(xm, lm), c='k',s=100, label='(xm, lm)')


# ------ problem 1 -------
x1_list = list()
l1_list = list()
z1_list = list()
for xi in x:
    tmp_list = [lagrangian(xi, li) for li in l]
    idx = np.argmax(tmp_list)
    x1_list.append(xi)
    l1_list.append(l[idx])
    z1_list.append(tmp_list[idx])
ax.scatter(x1_list, l1_list, z1_list, c='g', s=20, label='min_x max_{l>0}')

idx = np.argmin(z1_list)
print('min_x max_{l>0}')
print(f'(x, l) = ({x1_list[idx]}, {l1_list[idx]})')
print(f'p* = {z1_list[idx]}')


# ------ problem 2 -------
# z2_list = [lagrangian(xi, lm) for xi in x]
# ax.scatter(x, lm, z2_list, c='r', s=20, label='min L(x, lm)')


# ------ problem 3 -------
# fm_list = [lagrangian(xm, li) for li in l]
# ax.scatter(xm, l, fm_list, c='y', s=20, label='Grad L(x, l)')


# ------ problem 1 -------
x4_list = list()
l4_list = list()
z4_list = list()
for li in l:
    tmp_list = [lagrangian(xi, li) for xi in x]
    idx = np.argmin(tmp_list)
    x4_list.append(x[idx])
    l4_list.append(li)
    z4_list.append(tmp_list[idx])
ax.scatter(x4_list, l4_list, z4_list, c='b', s=20, label='max_{l>0} min_x')
idx = np.argmax(z4_list)
print('max_{l>0} min_x')
print(f'(x, l) = ({x4_list[idx]}, {l4_list[idx]})')
print(f'd* = {z4_list[idx]}')


ax.set_xlabel('x')
ax.set_ylabel('l')
ax.set_zlabel('L(x, l)')
plt.legend()
plt.show()


