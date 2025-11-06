import numpy as np
import matplotlib.pyplot as plt
# from altair import FontWeight

plt.rcParams["font.family"] = ["Times New Roman"]


# 手动输入 8x8 数据矩阵,热力图网格中对应的数据
data = np.array([
    [89.1, 88.5, 88.0, 87.5, 87.0, 86.5, 85.5, 84.5],
    [88.8, 88.2, 87.6, 87.0, 86.5, 86.0, 85.0, 84.0],
    [88.4, 87.8, 87.2, 86.8, 86.0, 85.5, 84.5, 83.8],
    [88.0, 87.5, 86.8, 86.0, 85.5, 84.8, 84.0, 83.5],
    [87.8, 87.0, 86.2, 85.5, 84.8, 84.2, 83.5, 83.0],
    [87.2, 86.6, 86.0, 85.2, 84.5, 83.8, 83.2, 82.5],
    [86.8, 86.2, 85.6, 84.8, 84.0, 83.5, 83.0, 82.0],
    [86.5, 85.8, 85.2, 84.5, 83.8, 83.0, 82.5, 82.0]
])

# x、y 轴刻度值
x_vals = [1,2,3,4,5,6,7,8]
y_vals = [1,2,3,4,5,6,7,8]

nx, ny = len(x_vals), len(y_vals)

# 绘图
fig, ax = plt.subplots(figsize=(10, 8)) #修改窗口大小

# 隐藏所有轴线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)


# 热力图
im = ax.imshow(data, aspect='auto', origin='lower', cmap='YlGnBu', vmin=80, vmax=90) #vmin、vmax控制色条表示的最小值和最大值

# 1. 自动从 data 查找最小值及其索引
score_val = data.min()  # 找到最小值 (82.0)
min_flat_idx = np.argmin(data)  # 找到最小值在扁平数组中的索引 (55)
# 将扁平索引转换回 2D 索引 (行, 列) -> (6, 7)
min_idx_y, min_idx_x = np.unravel_index(min_flat_idx, data.shape)

# 2. 根据索引从 x_vals 和 y_vals 中查找对应的 L 和 λ
# 注意：data[y, x] 对应 (y_vals[y], x_vals[x])
L_val = x_vals[min_idx_x]        # x_vals[7] -> 8
lambda_val = y_vals[min_idx_y]   # y_vals[6] -> 7

# 3. 格式化标题字符串 (使用 f-string)
title_text = f'Robot "purple" joint-3 with best params: {{L={L_val}, $\lambda$={lambda_val}}} and score: {score_val:.1f}'


# 色条
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', location='top',
                    shrink=1.0,  #控制色条长度
                    aspect=45, pad=0.01) #aspect控制色条上下高度

# 将 *动态生成* 的标题附加到 cbar.ax
cbar.ax.set_title(title_text, fontsize=18, fontweight='bold', pad=20)

# 设置色条刻度
cbar.ax.tick_params(labelsize=12)
plt.setp(cbar.ax.get_xticklabels(), fontweight='bold')

cbar.ax.xaxis.set_ticks_position('top')  #刻度放到上方
# 去掉色条边框
for spine in cbar.ax.spines.values():
    spine.set_visible(False)


# 坐标轴设置
ax.set_xticks(np.arange(nx))
ax.set_xticklabels(x_vals, fontsize=12, fontweight='bold')
ax.set_yticks(np.arange(ny))
ax.set_yticklabels(y_vals, fontsize=12, fontweight='bold')

# 'labelpad' 控制标签 ("L") 与刻度 (1, 2, 3...) 之间的距离
ax.set_xlabel("L", fontsize=15, fontweight='bold', labelpad=10)

# 'labelpad' 控制标签 ("λ") 与刻度 (8, 7, 6...) 之间的距离
ax.set_ylabel(r"$\lambda$", fontsize=15, fontweight='bold', labelpad=10)



# 在格子中显示分数
for i in range(ny):
    for j in range(nx):
        val = data[i, j]
        midpoint = (im.get_clim()[1] + im.get_clim()[0]) / 2
        text_color = "white" if val > midpoint else "black"  # 如果该格子的值比midpoint大，就让文字为白色；
        ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=12, color=text_color, fontweight='bold')

# 'pad' 参数控制图形边缘的空白大小
plt.tight_layout(pad=3.0)

plt.show()