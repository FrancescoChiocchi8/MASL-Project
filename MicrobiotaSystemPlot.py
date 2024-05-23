import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = pd.read_csv('output/MicrobiotaOutput_counts.csv')

fig, ax = plt.subplots()
ax.set_xlim(0, len(data))
ax.set_ylim(0, 600) 
plt.xlabel("Tick")
plt.ylabel("Agents") 

line_scfa, = ax.plot([], [], label='SCFA')
line_lps, = ax.plot([], [], label='LPS')
line_perm, = ax.plot([], [], label='Permeability')
line_cellEpit, = ax.plot([], [], label='cellEpit')

def init():
    line_scfa.set_data([], [])
    line_lps.set_data([], [])
    line_perm.set_data([], [])
    line_cellEpit.set_data([], [])
    return line_scfa, line_lps, line_perm, line_cellEpit

def update(frame):
    x = data['tick'].iloc[:frame+1]
    y_scfa = data['scfa'].iloc[:frame+1]
    y_lps = data['lps'].iloc[:frame+1]
    y_perm = data['permeability'].iloc[:frame+1]
    y_cellEpit= data['cellEpit'].iloc[:frame+1]

    line_scfa.set_data(x, y_scfa)
    line_lps.set_data(x, y_lps)
    line_perm.set_data(x, y_perm)
    line_cellEpit.set_data(x, y_cellEpit)

    return line_scfa, line_lps, line_perm, line_cellEpit

ani = FuncAnimation(fig, update, frames=len(data), init_func=init, blit=True)

ax.legend()

plt.title('Microbiota Environment Simulation')
plt.show()
