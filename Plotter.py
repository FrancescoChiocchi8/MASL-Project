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
    
    save_path = os.path.join(save_dir, f"{label.lower().replace(' ', '_')}_plot.png")
    count = 1
    while os.path.exists(save_path):
        save_path = os.path.join(save_dir, f"{label.lower().replace(' ', '_')}_plot_{count}.png")
        count += 1
        
    plt.savefig(save_path)
    plt.close()

def plot_agent_behavior(data_path, agent_types, title, output_path):
    data = pd.read_csv(data_path)
    ticks = data['tick']
    colors = plt.cm.tab20.colors
    for agent_type, color in zip(agent_types, colors):
        plt.plot(ticks, data[agent_type], label=agent_type, color=color)
    plt.xlabel('Tick')
    plt.ylabel('Agents')
    plt.title(title)
    plt.legend()
    output_folder = os.path.dirname(output_path)
    os.makedirs(output_folder, exist_ok=True)

    output_file = output_path
    count = 1
    while os.path.exists(output_file):
        base, ext = os.path.splitext(output_path)
        output_file = f"{base}_{count}{ext}"
        count += 1

    plt.savefig(output_file)
    plt.close()

def aggregate_plots(scenarios, save_path):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab20.colors

    for i, scenario in enumerate(scenarios):
        data = pd.read_csv(scenario['data_path'])
        ticks = data['tick']
        for j, agent_type in enumerate(scenario['agent_types']):
            color = colors[(i * len(scenario['agent_types']) + j) % len(colors)]
            plt.plot(ticks, data[agent_type], label=f"{scenario['title']} - {agent_type}", color=color)

    plt.xlabel('Tick')
    plt.ylabel('Agents')
    plt.title('All systems Visualization')
    plt.legend()
    plt.grid(True)

    count = 1
    while os.path.exists(save_path):
        base, ext = os.path.splitext(save_path)
        save_path = f"{base}_{count}{ext}"
        count += 1

    plt.savefig(save_path)
    plt.close()

data1 = pd.read_csv('output/MicrobiotaOutput_counts_12.csv')
data2 = pd.read_csv('output/LumeOutput_counts_12.csv')
data3 = pd.read_csv('output/NervousOutput_counts_12.csv')

plots1 = [
    {'y_col': 'scfa', 'label': 'SCFA', 'title': 'SCFA respect to Tick'},
    {'y_col': 'permeability', 'label': 'Permeability', 'title': 'Permeability respect to Tick'},
    {'y_col': 'lps', 'label': 'LPS', 'title': 'LPS respect to Tick'},
    {'y_col': 'cellEpit', 'label': 'cellEpit', 'title': 'cellEpit respect to Tick'},
    {'y_col': 'probioticArtificialAgent', 'label': 'PAA', 'title': 'PAA respect to Tick'}
]

plots2 = [
    {'y_col': 'lps', 'label': 'LPS', 'title': 'LPS respect to Tick'},
    {'y_col': 'tnfAlfa', 'label': 'TNF-Alpha', 'title': 'TNF-Alpha respect to Tick'},
    {'y_col': 'alfasin', 'label': 'alfasin', 'title': 'Alfasin respect to Tick'}
]

plots3 = [
    {'y_col': 'nadh', 'label': 'NADH', 'title': 'NADH respect to Tick'},
    {'y_col': 'ros', 'label': 'ROS', 'title': 'ROS respect to Tick'},
    {'y_col': 'alfasinucleina', 'label': 'Alfa-sinucleina', 'title': 'Alfa-sinucleina respect to Tick'},
    {'y_col': 'electron', 'label': 'Electron', 'title': 'Electron respect to Tick'},
    {'y_col': 'oxygen', 'label': 'Oxygen', 'title': 'Oxygen respect to Tick'}
]

scenarios = [
    {
        'data_path': 'output/NervousOutput_counts_12.csv',
        'agent_types': ['nadh', 'alfasinucleina', 'ros', 'artificialAgent', 'electron', 'oxygen'],
        'title': 'CNS - Behavior of the agents',
        'output_path': 'output/graphs/cns/all/cns_plot.png'
    },
    {
        'data_path': 'output/LumeOutput_counts_12.csv',
        'agent_types': ['lps', 'tnfAlfa', 'alfasin'],
        'title': 'LUME - Behavior of the agents',
        'output_path': 'output/graphs/lume/all/lume_plot.png'
    },
    {
        'data_path': 'output/MicrobiotaOutput_counts_12.csv',
        'agent_types': ['scfa', 'lps', 'permeability', 'cellEpit', 'probioticArtificialAgent'],
        'title': 'MICROBIOTA - Behavior of the agents',
        'output_path': 'output/graphs/microbiota/all/microbiota_plot.png'
    }
]

for plot in plots1:
    plot_data(data1, 'tick', plot['y_col'], plot['label'], color='blue', title=plot['title'], save_dir='output/graphs/microbiota')

for plot in plots2:
    plot_data(data2, 'tick', plot['y_col'], plot['label'], color='purple', title=plot['title'], save_dir='output/graphs/lume')

for plot in plots3:
    plot_data(data3, 'tick', plot['y_col'], plot['label'], color='green', title=plot['title'], save_dir='output/graphs/cns')

for scenario in scenarios:
    plot_agent_behavior(scenario['data_path'], scenario['agent_types'], scenario['title'], scenario['output_path'])

aggregate_plots(scenarios, 'output/graphs/all_plots_combined.png')
