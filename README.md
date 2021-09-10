# iCEM
improved Cross Entropy Method for trajectory optimization 

[presentation and experiments at https://martius-lab.github.io/iCEM/]

**Abstract:**
Trajectory optimizers for model-based reinforcement learning, such as the Cross-Entropy Method (CEM), can yield compelling results even in high-dimensional control tasks and sparse-reward environments. However, their sampling inefficiency prevents them from being used for real-time planning and control. We propose an improved version of the CEM algorithm for fast planning, with novel additions including temporally-correlated actions and memory, requiring 2.7-22x less samples and yielding a performance increase of 1.2-10x in high-dimensional control problems.
## Requirements
Installed via the provided Pipfile with `pipenv install`, then `pipenv shell` to activate virtualenv
## Running Experiments
- Inside icem folder run `python main.py settings/[env]/[json]`
- To render all envs: set `"render": true` in `iCEM/icem/settings/defaults/gt_default_env.json`


## iCEM improvements
The iCEM controller file is located [here](https://github.com/martius-lab/iCEM/blob/main/icem/controllers/icem.py) and it contains the following additions, which you can also extract and add to your codebase:
- **colored-noise**, [line68](https://github.com/martius-lab/iCEM/blob/8036c798a80afe8e821454ed68ad668d9499d7e0/icem/controllers/icem.py#L68):  
  It uses the package [`colorednoise`](https://pypi.org/project/colorednoise/) which generates `num_sim_traj` temporally correlated action sequences along the planning horizon dimension `h`.  
  The parameter you have to change depending on the task is `noise_beta` and it has an intuitive significance: higher β for low-frequency control (FETCH PICK&PLACE, RELOCATE, etc.) and lower β for high-frequency control (HALFCHEETAH RUNNING)  
  
&nbsp; | iCEM/CEM with ground truth | iCEM with PlaNet
:--- | :--- | :---
horizon h | 30 | 12
colored-noise exponent β | 0.25 HALFCHEETAH RUNNING | 0.25 CHEETAH RUN
&nbsp; |2.0 HUMANOID STANDUP | 0.25 CARTPOLE SWINGUP
&nbsp; |2.5 DOOR | 2.5 WALKER WALK
&nbsp; |2.5 DOOR (sparse reward) | 2.5 CUP CATCH
&nbsp; |3.0 FETCH PICK&PLACE | 2.5 REACHER EASY
&nbsp; |3.5 RELOCATE | 2.5 FINGER SPIN
 
- **clipping actions at boundaries**, [line79](https://github.com/martius-lab/iCEM/blob/8036c798a80afe8e821454ed68ad668d9499d7e0/icem/controllers/icem.py#L79):  
  Instead of sampling from a truncated normal distribution, we sample from the unmodified normal distribution (or colored-noise distribution) and clip the results
to lie inside the permitted action interval. This allows to sample maximal actions more frequently.

- **decay of population size**, [line126](https://github.com/martius-lab/iCEM/blob/8036c798a80afe8e821454ed68ad668d9499d7e0/icem/controllers/icem.py#L126):  
  Since the standard deviation of the CEM-distribution shrinks at every _CEM-iteration_, we introduce then an exponential decrease in population size of a fixed factor γ.: `num_sim_traj` now becomes `max(self.elites_size * 2, int(num_sim_traj / self.factor_decrease_num))`  
  The max operation ensures that the population size is at least double the elites' size.
  
- **keep previous elites**, [line143](https://github.com/martius-lab/iCEM/blob/8036c798a80afe8e821454ed68ad668d9499d7e0/icem/controllers/icem.py#L143):  
  We store the elite-set generated at each inner CEM-iteration and add a small fraction
of them (`fraction_elites_reused`) to the pool of the next iteration, instead of discarding the elite-set in each CEM-iteration.  
  
- **shift previous elites**, [line131](https://github.com/martius-lab/iCEM/blob/8036c798a80afe8e821454ed68ad668d9499d7e0/icem/controllers/icem.py#L131):  
  We store a small fraction of the elite-set of the last CEM-iteration and add each a random action at the end to use it in the next environment step.  
  This is done with the function [`elites_2_action_sequences`](https://github.com/martius-lab/iCEM/blob/8036c798a80afe8e821454ed68ad668d9499d7e0/icem/controllers/icem.py#L91).  
  The reason for not shifting the entire elite-set in both cases is that it would shrink the variance of CEM drastically in the first CEM-iteration because the last elites are quite likely dominating the new samples and have small variance. We use a `fraction_elites_reused`=0.3 in all experiments.  
  
- **execute best action**, [line163](https://github.com/martius-lab/iCEM/blob/8036c798a80afe8e821454ed68ad668d9499d7e0/icem/controllers/icem.py#L163):  
  The purpose of the original CEM algorithm is to estimate an unknown probability distribution. Using CEM as a trajectory optimizer detaches it from its original purpose. In the MPC context we are interested in the best possible action to be executed.  
  For this reason, we choose the first action of the best seen action sequence, rather than executing the first mean action, which was actually never evaluated.
  
- **add mean to samples** (at last iCEM-iteration), [line87](https://github.com/martius-lab/iCEM/blob/8036c798a80afe8e821454ed68ad668d9499d7e0/icem/controllers/icem.py#L87):  
  We decided to add the mean of the iCEM distribution as a sample for two reasons:  
    - because as dimensionality of the action space increases, it gets more and more difficult to sample an
action sequence closer to the mean of the distribution.  
    - because executing the mean might be beneficial for many tasks which require _“clean”_ action sequences like manipulation, object-reaching, or any linear trajectory in the state-space.  

  In practice, we add the mean just _at the last iteration_ for reasons explained in the paper, [section E.2](https://arxiv.org/pdf/2008.06389.pdf), and we simply substitute it to one of the samples: [`sampled_from_distribution[0] = self.mean`](https://github.com/martius-lab/iCEM/blob/8036c798a80afe8e821454ed68ad668d9499d7e0/icem/controllers/icem.py#L88)

## Improvements importance
In the figure below we present the ablations and additions of the improvements mentioned above, for all environments and a selection of budgets. As
we use the same hyperparameters for all experiments in some environments a few of the ablated versions perform slightly better but overall our final version has the best performance.  
  As we can see not all components are equally helpful in the different environments as each environment poses different challenges. For instance, in HUMANOID STANDUP the optimizer can get easily stuck in a local optimum corresponding to a sitting posture. Keeping balance in a standing position is also not trivial since small errors can lead to unrecoverable states. In the FETCH PICK&PLACE environment, on the other hand, the initial exploration is critical since the agent receives a meaningful
reward only if it is moving the box. Then colored noise and keep elites and shifting elites is most important.
<img width="800" alt="ablation_results" src="https://user-images.githubusercontent.com/29921252/106942629-b2bac400-6724-11eb-972a-906c67f9db16.png">


