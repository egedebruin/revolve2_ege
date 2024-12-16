import matplotlib.pyplot as plt
import numpy as np

# Sample data for 3 lines
x = np.linspace(0, 50, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)

# Plot the data
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.plot(x, y3, label='sin(x) * cos(x)')

# Choose a new x_value
x_value = 34

# Find corresponding y values at x=34 for each line
y1_value = np.interp(x_value, x, y1)
y2_value = np.interp(x_value, x, y2)
y3_value = np.interp(x_value, x, y3)

# Add a vertical line at x=34
plt.axvline(x=x_value, color='gray', linestyle='--')

# Annotate the y-values for each line
plt.annotate(f'{y1_value:.2f}', xy=(x_value, y1_value), xytext=(x_value+2, y1_value),
             arrowprops=dict(arrowstyle='->', color='blue'), color='blue')
plt.annotate(f'{y2_value:.2f}', xy=(x_value, y2_value), xytext=(x_value+2, y2_value),
             arrowprops=dict(arrowstyle='->', color='orange'), color='orange')
plt.annotate(f'{y3_value:.2f}', xy=(x_value, y3_value), xytext=(x_value+2, y3_value),
             arrowprops=dict(arrowstyle='->', color='green'), color='green')

# Add the current ticks
current_xticks = plt.gca().get_xticks()

# Add x_value=34 to the list of x-ticks
new_xticks = np.append(current_xticks, x_value)

# Set the new x-ticks with 34 included
plt.xticks(new_xticks)

# Add labels and a legend
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Show plot
plt.show()
