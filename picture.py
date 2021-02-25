import numpy as np 
import matplotlib.pyplot as plt 
# plot rank1
# 设置坐标轴的起始
plt.xlim(1,8)
plt.ylim(55,90)
# 画图
x = [1, 2, 4, 6, 8]
y1 = [58.49, 69.24, 75.93, 75.65, 75.31]
y2 = [71.18, 82.74, 86.45, 85.59, 85.73]
plt.plot(x,y1,'s-',color='r',label='Drone -> Satellite')
plt.plot(x,y2,'o-',color='b',label='Satellite -> Drone')
# plt.xlabel('Number of parts',weight='bold',fontsize=12)
# plt.ylabel('R@1(%)',weight='bold',fontsize=12)
# legend的位置
plt.legend(loc = 'lower right',edgecolor='black')
# 改变坐标字体的大小
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
# 设置x坐标轴的坐标
my_x_ticks = np.array([1,2,4,6,8])
plt.xticks(my_x_ticks)
# 显示网格
plt.grid(linestyle=':', linewidth=1.2, color='gray')
# 改变边框的显示形式
ax=plt.gca()
ax.spines['top'].set_linestyle(':')
ax.spines['right'].set_linestyle(':')
# 设置坐标轴线的宽度
ax.spines['top'].set_linewidth(1.2)
ax.spines['right'].set_linewidth(1.2)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
# 设置坐标轴的颜色
ax.spines['right'].set_color('grey')
ax.spines['top'].set_color('grey')
ax.spines['left'].set_color('grey')
ax.spines['bottom'].set_color('grey')
# 设置长宽比
ratio=1.5
ax.set_aspect(1.0/ax.get_data_ratio()*ratio)
# 保存
plt.savefig('/Users/wongtyu/Desktop/rank1.pdf')
# 显示
# plt.show()

# #plot mAP
# # 设置坐标轴的起始
# plt.xlim(1,8)
# plt.ylim(55,80)
# # 画图
# x = [1, 2, 4, 6, 8]
# y1 = [63.31, 72.91, 79.14, 78.86, 78.50]
# y2 = [58.74, 68.55, 74.79, 74.83, 74.35]
# plt.plot(x,y1,'s-',color='r',label='Drone -> Satellite')
# plt.plot(x,y2,'o-',color='b',label='Satellite -> Drone')
# # plt.xlabel('Number of parts n',weight='bold',fontsize=12)
# # plt.ylabel('AP(%)',weight='bold',fontsize=12)
# # legend的位置
# plt.legend(loc = 'lower right',edgecolor='black')
# # 改变坐标字体的大小
# plt.xticks(fontsize=11)
# plt.yticks(fontsize=11)
# # 设置x坐标轴的坐标
# my_x_ticks = np.array([1,2,4,6,8])
# plt.xticks(my_x_ticks)
# # 显示网格
# plt.grid(linestyle=':', linewidth=1.2, color='gray')
# # 改变边框的显示形式
# ax=plt.gca()
# ax.spines['top'].set_linestyle(':')
# ax.spines['right'].set_linestyle(':')
# # 设置坐标轴线的宽度
# ax.spines['top'].set_linewidth(1.2)
# ax.spines['right'].set_linewidth(1.2)
# ax.spines['left'].set_linewidth(1.2)
# ax.spines['bottom'].set_linewidth(1.2)
# # 设置坐标轴的颜色
# ax.spines['right'].set_color('grey')
# ax.spines['top'].set_color('grey')
# ax.spines['left'].set_color('grey')
# ax.spines['bottom'].set_color('grey')
# # 设置长宽比
# ratio=1.5
# ax.set_aspect(1.0/ax.get_data_ratio()*ratio)
# # 保存
# plt.savefig('/Users/wongtyu/Desktop/AP.pdf')
# 显示
# plt.show()