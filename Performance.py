import psutil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Inizializza le liste per memorizzare i dati
cpu_usage = []
memory_usage = []

# Funzione per aggiornare i dati
def update_data(frame):
    cpu_percent = psutil.cpu_percent()
    mem_percent = psutil.virtual_memory().percent
    
    cpu_usage.append(cpu_percent)
    memory_usage.append(mem_percent)
    
    # Limita le liste ai 50 elementi più recenti per evitare sovraccarichi di memoria
    if len(cpu_usage) > 50:
        cpu_usage.pop(0)
        memory_usage.pop(0)
    
    # Pulisce il grafico
    plt.clf()
    
    # Plotta i dati
    plt.subplot(3, 1, 1)
    plt.plot(cpu_usage, label='CPU')
    plt.ylabel('CPU (%)')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(memory_usage, label='Memory')
    plt.ylabel('Memory (%)')
    plt.legend()

# Crea un'animazione che aggiorna i dati ogni 1000 millisecondi
ani = FuncAnimation(plt.gcf(), update_data, interval=1000)
# Visualizza il grafico
plt.show()
#plt.savefig('performance_plot.png')