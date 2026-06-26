import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据
lambda2_values = np.array([0.1, 1, 5, 10, 20, 50])
lambda1_values = np.array([0.1, 1, 5, 10, 20, 50])

# 性能数据 [Lambda1, Lambda2]
data_acc = np.array([
    [90.76, 98.00, 71.64, 66.16, 71.88, 58.80],
    [98.80, 98.96, 98.60, 98.56, 97.32, 82.00],
    [98.72, 98.88, 99.04, 98.96, 98.60, 98.52],
    [98.60, 98.84, 98.92, 99.04, 98.96, 98.60],
    [98.60, 98.76, 98.92, 98.92, 99.04, 98.84],
    [98.60, 98.68, 98.80, 98.88, 98.92, 99.08]
])

data_nmi = np.array([
    [85.25, 94.69, 69.93, 41.59, 55.66, 42.59],
    [96.42, 96.77, 95.96, 95.88, 92.51, 72.95],
    [96.10, 96.55, 96.94, 96.74, 95.96, 95.79],
    [95.99, 96.43, 96.66, 96.94, 96.74, 95.96],
    [95.99, 96.34, 96.69, 96.66, 96.94, 96.52],
    [95.99, 96.02, 96.29, 96.59, 96.66, 97.06]
])

data_ari = np.array([
    [81.09, 95.11, 60.67, 35.27, 46.33, 32.18],
    [97.03, 97.43, 96.55, 96.45, 93.47, 67.45],
    [96.84, 97.23, 97.62, 97.42, 96.55, 96.35],
    [96.55, 97.13, 97.33, 97.62, 97.42, 96.55],
    [96.55, 96.94, 97.33, 97.33, 97.62, 97.13],
    [96.55, 96.74, 97.03, 97.23, 97.33, 97.72]
])

# 彩虹配色 & 透明度
colors_list = ['#008feb', '#3dbd84', '#47d6d4', '#ffd155', '#ff8470', '#ce8bdb']  

def plot_3d_bars(ax, data, title, z_label, colors_list):
    """绘制3D柱状图 - 沿lambda2方向每行颜色相同"""
    x = np.arange(len(lambda2_values))
    y = np.arange(len(lambda1_values))
    xx, yy = np.meshgrid(x, y)
    
    # 展平数据
    x_pos = xx.flatten()
    y_pos = yy.flatten()
    z_pos = np.zeros_like(x_pos)
    
    # 柱子尺寸 - 几乎无缝隙
    dx = dy = 0.95
    
    # 高度数据
    dz = data.flatten()
    
    # 根据lambda1的索引设置颜色（每行颜色相同）
    bar_colors = []
    for i in range(len(x_pos)):
        row_idx = int(y_pos[i])  # 获取所在的行（lambda1索引）
        bar_colors.append(colors_list[row_idx])
    
    # 绘制3D柱状图 - 卡通风格
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=bar_colors, 
             edgecolor='white', linewidth=1, alpha=0.95)
    
    # 设置标签
    ax.set_xlabel(r'$\lambda_2$', fontsize=12, labelpad=8, fontweight='bold')
    ax.set_ylabel(r'$\lambda_1$', fontsize=12, labelpad=8, fontweight='bold')
    ax.set_zlabel(z_label, fontsize=12, labelpad=12, fontweight='bold')
    
    # 设置刻度
    ax.set_xticks(np.arange(len(lambda2_values)))
    ax.set_yticks(np.arange(len(lambda1_values)))
    ax.set_xticklabels([f'{l}' for l in lambda2_values], fontsize=10)
    ax.set_yticklabels([f'{l}' for l in lambda1_values], fontsize=10)
    
    # 设置标题
    # ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color='#2C3E50')
    
    # 调整视角
    ax.view_init(elev=25, azim=-60)
    
    # 设置z轴范围
    ax.set_zlim([0, 110])
    
    # 设置背景色
    ax.set_facecolor('#F8F9FA')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

# 创建图形 - 卡通风格
fig = plt.figure(figsize=(14, 4.5), facecolor='#F8F9FA')

# 绘制ACC
ax1 = fig.add_subplot(131, projection='3d')
plot_3d_bars(ax1, data_acc, 'ACC', 'ACC (%)', colors_list)

# 绘制NMI
ax2 = fig.add_subplot(132, projection='3d')
plot_3d_bars(ax2, data_nmi, 'NMI', 'NMI (%)', colors_list)

# 绘制ARI
ax3 = fig.add_subplot(133, projection='3d')
plot_3d_bars(ax3, data_ari, 'ARI', 'ARI (%)', colors_list)



plt.savefig('BDGP_lamba_cartoon.png', dpi=300,  facecolor='#F8F9FA', pad_inches=0.1)
plt.savefig('BDGP_lamba_cartoon.pdf', dpi=300,  facecolor='#F8F9FA', pad_inches=0.1)

plt.show()
# 打印最佳参数
print("=" * 60)
print("BDGP50 最佳性能参数分析")
print("=" * 60)

for data, metric in zip([data_acc, data_nmi, data_ari], ['ACC', 'NMI', 'ARI']):
    max_idx = np.unravel_index(np.argmax(data), data.shape)
    max_val = data[max_idx]
    print(f"\n{metric}:")
    print(f"  最佳值: {max_val:.2f}%")
    print(f"  对应参数: λ1 = {lambda1_values[max_idx[0]]}, λ2 = {lambda2_values[max_idx[1]]}")

print("\n" + "=" * 60)