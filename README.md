# iCEM
improved Cross Entropy Method for trajectory optimization

**Abstract:**
Trajectory optimizers for model-based reinforcement learning, such as the Cross-Entropy Method (CEM), can yield compelling results even in high-dimensional control tasks and sparse-reward environments. However, their sampling inefficiency prevents them from being used for real-time planning and control. We propose an improved version of the CEM algorithm for fast planning, with novel additions including temporally-correlated actions and memory, requiring 2.7-22x less samples and yielding a performance increase of 1.2-10x in high-dimensional control problems.
## Requirements
Installed via the provided Pipfile with `pipenv install`, then `pipenv shell` to activate virtualenv
## Running Experiments
Inside icem folder run `python icem.py settings/[env]/[json]`

