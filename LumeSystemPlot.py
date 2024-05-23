import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = pd.read_csv('output/LumeOutput_counts.csv')

fig, ax = plt.subplots()
ax.set_xlim(0, len(data))
ax.set_ylim(0, 600) 
plt.xlabel("Tick")
plt.ylabel("Agents")

line_tnf, = ax.plot([], [], label='TNF-alpha')
line_lps, = ax.plot([], [], label='LPS')
line_alpha, = ax.plot([], [], label='alpha-sinucleina')


def init():
    line_tnf.set_data([], [])
    line_lps.set_data([], [])
    line_alpha.set_data([], [])
    return line_tnf, line_lps, line_alpha

def update(frame):
    x = data['tick'].iloc[:frame+1]
    y_scfa = data['tnfAlfa'].iloc[:frame+1]
    y_lps = data['lps'].iloc[:frame+1]
    y_perm = data['alfasin'].iloc[:frame+1]

    line_tnf.set_data(x, y_scfa)
    line_lps.set_data(x, y_lps)
    line_alpha.set_data(x, y_perm)

    return line_tnf, line_lps, line_alpha

ani = FuncAnimation(fig, update, frames=len(data), init_func=init, blit=True)

ax.legend()

plt.title('Lume Environment Simulation')
plt.show()