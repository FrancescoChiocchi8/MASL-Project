import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = pd.read_csv('output/NervousOutput_counts.csv')

fig, ax = plt.subplots()
ax.set_xlim(0, len(data))
ax.set_ylim(0, 1000)  
plt.xlabel("Tick")
plt.ylabel("Agents")

line_nadh, = ax.plot([], [], label='nadh')
line_alfasinucleina, = ax.plot([], [], label='alfasinucleina')
line_ros, = ax.plot([], [], label='ros')
line_electron, = ax.plot([], [], label='electron')
line_oxygen, = ax.plot([], [], label='oxygen')

def init():
    line_nadh.set_data([], [])
    line_alfasinucleina.set_data([], [])
    line_ros.set_data([], [])
    line_electron.set_data([], [])
    line_oxygen.set_data([], [])
    return line_nadh, line_alfasinucleina, line_ros, line_electron, line_oxygen

def update(frame):
    x = data['tick'].iloc[:frame+1]
    y_nadh = data['nadh'].iloc[:frame+1]
    y_alfasinucleina = data['alfasinucleina'].iloc[:frame+1]
    y_ros = data['ros'].iloc[:frame+1]
    y_electron= data['electron'].iloc[:frame+1]
    y_oxygen= data['oxygen'].iloc[:frame+1]

    line_nadh.set_data(x, y_nadh)
    line_alfasinucleina.set_data(x, y_alfasinucleina)
    line_ros.set_data(x, y_ros)
    line_electron.set_data(x, y_electron)
    line_oxygen.set_data(x, y_oxygen)

    return line_nadh, line_alfasinucleina, line_ros, line_electron, line_oxygen

ani = FuncAnimation(fig, update, frames=len(data), init_func=init, blit=True)

ax.legend()

plt.title('Nervous System Environment Simulation')
plt.show()