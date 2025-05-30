# Modular Pipeline for Model‑Based Reinforcement Learning: Dyna and MPC with Shooting

## 1  Scope and Objectives
1. Develop a reusable codebase that implements two model‑based RL approaches:
Dyna (planning with a learned dynamics model)
Model Predictive Control (MPC) with a shooting‑method optimizer
Support multiple task (environment) types
At least one continuous‑state, continuous‑action environment (e.g., Inverted Pendulum‑v2, HalfCheetah‑v4)
At least one discrete‑action environment (e.g., CartPole‑v1, LunarLander‑v2)
Expose modular choices (You allow this in your main file)
Environment selection
Dynamics‑model architecture selection (MLP, RNN, ensemble, etc.)
Control‑algorithm selection (Dyna or MPC)
Ensure full reproducibility
Deterministic seeding, logging, and automatic result‑folder creation
Clear installation and run scripts that work on a fresh machine
## 2  Deliverables and Deadlines
A. Repository skeleton (end of next week)

Structure must include mbrl_pipeline/, algorithms/, models/, envs/, scripts/, results/, tests/, docs/.

B. Running example (end of next week)

One example run—for example, Dyna + CartPole with an MLP dynamics model—must train, evaluate, and save results automatically.

C. README.md (due with running example)

Must contain project overview, quick‑start commands, brief algorithm summaries, bibliography, and troubleshooting tips.

D. Setup files (due with running example)

Provide either

environment.yml (Conda) or
pyproject.toml plus requirements.txt.
Must install cleanly on Ubuntu 22.04 and macOS 14 in ≤ 10 min.
E. Logging & outputs (due with running example)

Code must automatically create results/<env>/<algorithm>/<timestamp>/ containing metrics, trained models, and figures.

AI assistance is permitted and encouraged. Generated code must be reviewed, cleaned, and fully understood before submission.

# 3  Repository Structure (example)
mbrl_pipeline/

├── algorithms/

│   ├── dyna.py

│   ├── mpc_shooting.py

├── models/

│   ├── mlp_dynamics.py

│   ├── rnn_dynamics.py

├── envs/

│   ├── cartpole.py

│   └── pendulum.py

├── scripts/

│   ├── train.py

│   └── evaluate.py

├── results/           # auto‑generated; keep empty in repo

├── tests/

│   └── test_smoke.py

├── docs/

│   └── algorithm_overview.md

├── README.md

├── pyproject.toml      # or setup.cfg / setup.py/ or a pickle file

└── requirements.txt


# 4  Implementation Requirements

Frameworks
PyTorch ≥ 2.2, Gymnasium ≥ 0.29, NumPy ≥ 1.26
Optional for MPC: CasADi or JAX (justify choice in README)
Configuration management
 Use argparse to select components, e.g.
        
python scripts/train.py --env CartPole-v1 --algo dyna --model mlp --seed 42


Logging

 TensorBoard or Weights & Biases for scalars. Always write JSON/CSV copies.
Reproducibility

Set seeds for torch, numpy, and gymnasium.

 Store complete hyper‑parameters in a config.yaml inside each results folder.
Testing

Provide at least one unit test per major module
# 5  Key References
Dyna: Sutton, R. S. (1990). “Integrated Architectures for Learning, Planning, and Reacting.” ICML.  Sutton & Barto (2020). Reinforcement Learning: An Introduction (2nd ed.), Chapter 8.

Model learning: Deisenroth & Rasmussen (2011). “PILCO: A Model‑Based and Data‑Efficient RL Method.”  Chua et al. (2018). “Deep RL in a Handful of Trials with Probabilistic Ensembles.” https://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf?utm_source=chatgpt.com; https://arxiv.org/abs/1805.12114?utm_source=chatgpt.com  

MPC + shooting: Nagabandi et al. (2018). “Neural Network Dynamics for Model‑Based Deep RL with MPC.”  Kamthe & Deisenroth (2018). “Data‑efficient RL with Probabilistic MPC.” https://people.eecs.berkeley.edu/~ronf/PAPERS/anagabandi-icra18.pdf?utm_source=chatgpt.com;  https://proceedings.mlr.press/v84/kamthe18a/kamthe18a.pdf?utm_source=chatgpt.com 

Survey: Moerland et al. (2023). “Model‑Based Reinforcement Learning: A Survey.” Foundations and Trends in Machine Learning. https://arxiv.org/abs/2006.16712?utm_source=chatgpt.com 

# 6  Milestone Checklist (due next week)
Repository cloned by all team members.
pip install -r requirements.txt (or conda env create -f environment.yml) runs without errors.

python scripts/train.py completes one full training session on CartPole in ≤ 10 min, and also one full training session on Pendulum.

Reward curve saved to results/.../reward.png.

README quick‑start verified by a peer not involved in coding.




