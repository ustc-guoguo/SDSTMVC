import matplotlib.pyplot as plt
import numpy as np

# 1. 全局字体与格式设置 (设置 Times New Roman 和支持数学公式渲染)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # 使得公式中的字体与 Times New Roman 风格一致
plt.rcParams['axes.linewidth'] = 1.5       # 设置坐标轴边框的粗细

# 2. 生成 Mock 数据 (根据原图趋势模拟)
# x 轴坐标: 1 到 10，代表 2^1 到 2^10
x = np.arange(1, 11) 

# 四个指标的性能数据
acc = [0.55, 0.53, 0.64, 0.68, 0.74, 0.75, 0.74, 0.755, 0.752, 0.758]
nmi = [0.49, 0.48, 0.55, 0.58, 0.60, 0.61, 0.60, 0.605, 0.608, 0.61]
pur = [0.56, 0.54, 0.64, 0.68, 0.74, 0.755, 0.74, 0.755, 0.752, 0.758]
ari = [0.37, 0.34, 0.43, 0.47, 0.52, 0.53, 0.52, 0.54, 0.535, 0.54]

# 3. 创建画布
fig, ax = plt.subplots(figsize=(8, 5))

# 4. 绘制折线图
# ACC: 绿色实线 + 圆圈
ax.plot(x, acc, color='#189E1D', linestyle='-', marker='o', linewidth=2.5, markersize=7, label='ACC')
# NMI: 蓝色虚线 + 方块
ax.plot(x, nmi, color='#001CD1', linestyle='--', marker='s', linewidth=2.5, markersize=6, label='NMI')
# PUR: 橙色虚线 + 五角星
ax.plot(x, pur, color='#FFA500', linestyle='--', marker='*', linewidth=2.5, markersize=10, label='PUR')
# ARI: 红色虚线 + 圆圈
ax.plot(x, ari, color='#E31A1C', linestyle='--', marker='o', linewidth=2.5, markersize=7, label='ARI')

# 5. 坐标轴范围与刻度设置
ax.set_ylim(0.0, 1.0)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

ax.set_xlim(0.5, 10.5)
ax.set_xticks(x)
# 使用 LaTeX 语法生成 x 轴标签 (例如: 2^1, 2^2)
ax.set_xticklabels([f'$2^{{{i}}}$' for i in x], fontsize=15)

# 6. 设置坐标轴标签
ax.set_xlabel('Feature Dimensions', fontsize=16, fontweight='bold')
ax.set_ylabel('Metrics', fontsize=16, fontweight='bold')

# 7. 刻度线样式微调 (朝内，加粗)
ax.tick_params(axis='both', which='major', labelsize=14, direction='in', length=6, width=1.5)

# 8. 添加极细的辅助网格线 (可选，原图似乎有极淡的背景网格)
ax.grid(True, linestyle=':', alpha=0.4)

# 9. 顶部图例设置
# bbox_to_anchor 把图例定在图表外部正上方，ncol=4 保证横向排列
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=4, 
                   prop={'size': 14, 'weight': 'bold'}, frameon=True)
# 让图例边框颜色变淡，还原原图的精致感
legend.get_frame().set_edgecolor('#CCCCCC')
legend.get_frame().set_linewidth(1)

# 10. 调整布局并保存
plt.tight_layout()
# 存为高质量 PDF (顶会强烈推荐) 或 PNG
plt.savefig('feature_dimensions_metrics.pdf', dpi=300, bbox_inches='tight')
plt.savefig('feature_dimensions_metrics.png', dpi=300, bbox_inches='tight')

plt.show()