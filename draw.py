import matplotlib.pyplot as plt

# 示例数据
x_values = [2, 4, 8, 16]
y_values = [36.66, 33.75, 34.83, 42.61]

# 创建折线图
plt.plot(x_values, y_values, label='time-line_number')

# 添加标题和标签
plt.title('process effect')
plt.xlabel('line_number ')
plt.ylabel('time ')

# 添加图例
plt.legend()

# 显示图形
plt.show()
