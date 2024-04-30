import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_data(data, x_col, y_col, label, color, linestyle='-', marker='o', title=None, save_dir='output/graphs'):
    plt.figure(figsize=(10, 6))
    plt.plot(data[x_col], data[y_col], linestyle=linestyle, marker=marker, color=color, label=label)
    plt.xlabel('Tick')
    plt.ylabel(label)
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)

    if 'microbiota' in save_dir.lower():
        save_path = os.path.join(save_dir, f"microbiota_{label.lower().replace(' ', '_')}_plot.png")
    elif 'cns' in save_dir.lower():
        save_path = os.path.join(save_dir, f"cns_{label.lower().replace(' ', '_')}_plot.png")
    else:
        save_path = os.path.join(save_dir, f"lume_{label.lower().replace(' ', '_')}_plot.png")

    count = 1
    while os.path.exists(save_path):
        if 'microbiota' in save_dir.lower():
            save_path = os.path.join(save_dir, f"microbiota_{label.lower().replace(' ', '_')}_plot_{count}.png")
        elif 'cns' in save_dir.lower():
            save_path = os. path.join(save_dir, f"cns_{label.lower().replace(' ', '_')}_plot_{count}.png")
        else:
            save_path = os.path.join(save_dir, f"lume_{label.lower().replace(' ', '_')}_plot_{count}.png")
        count += 1

    plt.savefig(save_path)
    plt.close()

def plot_agent_behavior(data_path, agent_types, title, output_path):
    data = pd.read_csv(data_path)

    ticks = data['tick']

    for agent_type in agent_types:
        plt.plot(ticks, data[agent_type], label=agent_type)

    plt.xlabel('Tick')
    plt.ylabel('Agents')
    plt.title(title)
    plt.legend()

    output_folder = os.path.dirname(output_path)
    os.makedirs(output_folder, exist_ok=True)

    plt.savefig(output_path)

    plt.close()

data1 = pd.read_csv('output/MicrobiotaOutput_counts.csv')
data2 = pd.read_csv('output/lumeOutput_counts.csv')
data3 = pd.read_csv('output/agent_counts.csv')

plots1 = [
    {'y_col': 'scfa', 'label': 'SCFA', 'color': 'blue', 'title': 'SCFA respect to Tick'},
    {'y_col': 'permeability', 'label': 'Permeability', 'color': 'green', 'title': 'Permeability respect to Tick'},
    {'y_col': 'lps', 'label': 'LPS', 'color': 'red', 'title': 'LPS respect to Tick'}
]

plots2 = [
    {'y_col': 'lps', 'label': 'LPS', 'color': 'purple', 'title': 'LPS respect to Tick'},
    {'y_col': 'tnfAlfa', 'label': 'TNF-Alpha', 'color': 'orange', 'title': 'TNF-Alpha respect to Tick'}
]

plots3 = [
    {'y_col': 'nadh', 'label': 'NADH', 'color': 'blue', 'title': 'NADH respect to Tick'},
    {'y_col': 'ros', 'label': 'ROS', 'color': 'red', 'title': 'ROS respect to Tick'},
    {'y_col': 'alfasinucleina', 'label': 'Alfa-sinucleina', 'color': 'green', 'title': 'Alfa-sinucleina respect to Tick'},
    {'y_col': 'electron', 'label': 'Electron', 'color': 'purple', 'title': 'Electron respect to Tick'},
    {'y_col': 'oxygen', 'label': 'Oxygen', 'color': 'orange', 'title': 'Oxygen respect to Tick'}
]

scenarios = [
    {
        'data_path': 'output/agent_counts.csv',
        'agent_types': ['nadh', 'alfasinucleina', 'ros', 'artificialAgent', 'electron', 'oxygen'],
        'title': 'CNS - Behavior of the agents',
        'output_path': 'output/graphs/cns/all/cns_plot.png'
    },
    {
        'data_path': 'output/lumeOutput_counts.csv',
        'agent_types': ['lps', 'tnfAlfa'],
        'title': 'LUME - Behavior of the agents',
        'output_path': 'output/graphs/lume/all/lume_plot.png'
    },
    {
        'data_path': 'output/MicrobiotaOutput_counts.csv',
        'agent_types': ['scfa', 'lps', 'permeability', 'cellEpit'],
        'title': 'MICROBIOTA - Behavior of the agents',
        'output_path': 'output/graphs/microbiota/all/microbiota_plot.png'
    }
]

for plot in plots1:
    plot_data(data1, 'tick', plot['y_col'], plot['label'], plot['color'], title=plot['title'], save_dir='output/graphs/microbiota')

for plot in plots2:
    plot_data(data2, 'tick', plot['y_col'], plot['label'], plot['color'], title=plot['title'], save_dir='output/graphs/lume')

for plot in plots3:
    plot_data(data3, 'tick', plot['y_col'], plot['label'], plot['color'], title=plot['title'], save_dir='output/graphs/cns')

for scenario in scenarios:
    plot_agent_behavior(scenario['data_path'], scenario['agent_types'], scenario['title'], scenario['output_path'])
