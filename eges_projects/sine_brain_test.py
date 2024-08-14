import matplotlib.pyplot as plt
import math

dt = 0.05
frequency = 4
phase = 3
amplitude = 1
min_touch = 50
max_touch = 100

fig, ax = plt.subplots(nrows=8, sharex=True, sharey=True)
for j, touch_weight in enumerate(range(-8, 0)):
    for (color, sensor_phase_offset, label) in [('red', 0, '0'), ('blue', 3, 'pi')]:
        t = phase
        xs = []
        ys = []
        for i in range(150):
            # if min_touch <= i < max_touch:
            #     touch_sensor = 1
            #     ax[j].scatter(i * dt, 1, color='black', s=1)
            #     ax[j].scatter(i * dt, -1, color='black', s=1)
            # else:
            #     touch_sensor = 0
            # if i == min_touch or i == max_touch:
            #     for k in range(-10, 11):
            #         ax[j].scatter(i * dt, k/10, color='black', s=1)
            target = amplitude * math.sin(t)

            touch_sensor = 1

            t += dt * frequency + dt * touch_sensor * touch_weight * math.cos(t + sensor_phase_offset)
            xs.append(i*dt)
            ys.append(target)

        ax[j].plot(xs, ys, color=color, label=label)
        ax[j].title.set_text(f"Touch weight: {touch_weight}")
        if j == 0:
            ax[j].legend(loc='upper left')
plt.show()

