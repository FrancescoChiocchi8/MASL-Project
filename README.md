# MASL-Project
Repository for Multi Agent System Lab project

## Info
This study aims to investigate and model this hypothesized pathway of the microbiota-intestine-brain axis using computational simulation.
Using Repast4Py, an agent-based modeling platform, our goal is to simulate the dynamics of α-synuclein aggregation and transmission within the intestinal lumen, and to observe its impact on the CNS. Through this simulation, we aim to gain insights into the potential role of intestinal α-synuclein in PD pathogenesis and to explore novel therapeutic strategies targeting the microbiota-intestine-brain axis.


## Usage

To get started with this project, follow this [guide](https://repast.github.io/repast4py.site/guide/user_guide.html#_getting_started) to download all necessary dependencies and tools.

After completing the initial setup steps, you can proceed with the following instructions:

1. Open your terminal or command prompt.

2. Clone the repository to your local machine:

```bash
 git clone https://github.com/FrancescoChiocchi8/MASL-Project
```

3. Install requirements:

```bash
 pip install -r requirements.txt
```

4. To run the application and generate [output](/output), use the following command:
```bash
 mpirun -n 4 python3 GastroIntestinalSystem.py GastroIntestinalSpec.yaml
```

## Graphical View

- To run the GUI, use the following command:
```bash
 python3 GUI.py
```
Press 'a' for automatic mode, 'm' for manual mode and stop iteration, left and right arrow for change tick view.

- To generate graphs, run the following command. The output graphs will be stored in the [graphs directory](/output/graphs/):
```bash
 python3 Plotter.py
```

- To run animation for the system (microbiota, lumen, cns), run the following command:
```bash
 python3 Animation.py
```