---
layout: single
# classes: wide
author_profile: false
permalink: /disropt
title: "DISROPT"
defaults:
  - scope:
      path: ""
      type: pages
    values:
toc: true
toc_sticky: true
---

## Introduction to DISROPT


**DISROPT** is a Python package for distributed optimization over peer-to-peer networks of computing units called agents.


### Objective functions and constraints


**DISROPT** comes with many already implemented mathematical functions.
Functions are defined in terms of optimization variables (`Variable`) or other functions. A variable in $$x\in\mathbb{R}^{2}$$ can be defined as


    from disropt.functions import Variable
    n = 2 # dimension of the variable
    x = Variable(n)


Now, suppose you want to define an affine function $$f(x)=A^\top x - b$$ with $$A\in\mathbb{R}^{2\times 2}$$ and $$b\in\mathbb{R}^2$$


    import numpy as np
    a = 1
    A = np.array([[1,2], [2,4]])
    b = np.array([[1], [1]])
    f = A @ x - b

Constraints are represented in the canonical forms $$f(x)=0$$ and $$f(x)\leq 0$$.

They are directly obtained from functions::

    constraint = g == 0 # g(x) = 0
    constraint = g >= 0 # g(x) >= 0
    constraint = g <= 0 # g(x) <= 0

On the right side of (in)equalities, numpy arrays and functions (with appropriate shapes) are also allowed::

    c = np.random.rand(2,1)
    constr = f <= c

which is automatically translated in the corresponding canonical form.



### Optimization Problems

The `Problem` class allows one to define optimization problems of various types. Consider the following problem:

$$\begin{eqnarray}
\text{minimize } & \| A^\top x - b \| \\
    \text{subject to } & x \geq 0
\end{eqnarray}$$

with $$x\in\mathbb{R}^4$$. We can define it as

    import numpy as np
    from disropt.problems import Problem
    from disropt.functions import Variable, Norm

    x = Variable(4)
    A = np.random.randn(n, n)
    b = np.random.randn(n, 1)

    obj = Norm(A @ x - b)
    constr = x >= 0

    pb = Problem(objective_function = obj, constraints = constr)

In the distributed framework of **DISROPT**, the `Problem` class is mainly meant to define portions of a bigger optimization problem that are locally known to local computing units. However, since in many distributed optimization algorithms can be requested to solve local optimization problems, we implemented some problem solvers. If the problem is convex, it can be solved as

    solution = pb.solve()

Generic (convex) nonlinear problems of the form

$$\begin{eqnarray}

    \text{minimize } & f(x) \\

    \text{subject to } & g(x) \leq 0 \\

                        & h(x) = 0
\end{eqnarray}$$

are solved through the cvxpy solver (when possible), or with the cvxopt solver, while more structured problems (LPs and QPs) can be solved through other solvers (osqp and glpk). Mixed-Integer Problems can be solved using gurobi.

### Agents in the network

The `Agent` class is meant to represent the local computing units that collaborate in the network in order to solve some specific problem.

Agents are instantiated by defining their in/out-neighbors and the weights they assign their neighbors. For example, consider the following network

<center>
<img src="/images/network.png">
</center>

Then, agent 0 is defined as

    from disropt.agents import Agent

    agent = Agent(in_neighbors=[1,2],
                  out_neighbors=[2],
                  weights=[0.3, 0.2])



Assigning a local optimization problem to an agent is done via the `set_problem` method, which modifies the `problem` attribute of the agent. Consider the following code

     from disropt.problems import Problem
     from disropt.functions import Variable, Norm
     x = Variable(4)
     A = np.random.randn(n, n)
     b = np.random.randn(n, 1)
     obj = Norm(A @ x - b)
     constr = x >= 0
     pb = Problem(objective_function = obj, constraints = constr)

Assume that the variable `problem` contains the local problem data. Then, the variable is assigned to the agent by

    agent.set_problems(problem)

### Distributed Optimization Algorithms

In **DISROPT**, there are many implemented distributed optimization algorithms. Each algorithm is tailored
for a specific distributed optimization set-up. We refer the reader to the [DISROPT documentation page](https://disropt.readthedocs.io/en/latest/index.html) for a detailed explanation.


## Constraint-coupled: charging of Plug-in Electric Vehicles (PEVs)


We now consider the problem of determining an optimal overnight charging schedule for a fleet of Plug-in Electric Vehicles (PEVs).

### Problem formulation

Suppose there is a fleet of `N` PEVs (agents) that must be charged by drawing power from
the same electricity distribution network. Assuming the vehicles are connected to the grid
at a certain time (e.g., at midnight), the goal is to determine an optimal overnight schedule
to charge the vehicles, since the electricity price varies during the charging period.

Formally, we divide the entire charging period into a total of :math:`T = 24` time slots,
each one of duration :math:`\Delta T = 20` minutes. For each PEV :math:`i \in \{1, \ldots, N\}`,
the charging power at time step :math:`k` is equal to :math:`P_i u_i(k)`, where :math:`u_i(k) \in [0, 1]`
is the input to the system and :math:`P_i` is the maximum charging power that can be fed to the
:math:`i`-th vehicle.

System model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The state of charge of the :math:`i`-th battery is denoted by :math:`e_i(k)`,
its initial state of charge is :math:`E_i^\text{init}`, which by the end of the charging
period has to attain at least :math:`E_i^\text{ref}`. The charging conversion efficiency
is denoted by :math:`\zeta_i^\text{u} \triangleq 1 - \zeta_i`, where :math:`\zeta_i > 0`
encodes the conversion losses. The battery's capacity limits are denoted by
:math:`E_i^\text{min}, E_i^\text{max} \ge 0`. The system's dynamics are therefore given by

.. math::

  & \: e_i(0) = E_i^\text{init}
  \\
  & \: e_i(k+1) = e_i(k) + P_i \Delta T \zeta_i^u u_i(k), \hspace{0.5cm} k \in \{0, \ldots, T-1\}
  \\
  & \: e_i(T) \ge E_i^\text{ref}
  \\
  & \: E_i^\text{min} \le e_i(k) \le E_i^\text{max}, \hspace{2.88cm} k \in \{1, \ldots, T\}
  \\
  & \: u_i(k) \in [0,1], \hspace{4.47cm} k \in \{0, \ldots, T-1\}.

To model congestion avoidance of the power grid, we further consider the following (linear)
coupling constraints among all the variables

.. math::

  \sum_{i=1}^N P_i u_i(k) \le P^\text{max}, \hspace{1cm} k \in \{0, \ldots, T-1\},

where :math:`P^\text{max}` is the maximum power that the be drawn from the grid.

Optimization problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We assume that, at each time slot :math:`k`, electricity has unit cost equal to
:math:`C^\text{u}(k)`. Since the goal is to minimize the overall consumed energy price,
the global optimization problem can be posed as

.. math::

  \min_{u, e} \: & \: \sum_{i=1}^N \sum_{k=0}^{T-1} C^\text{u}(k) P_i u_i(k)
  \\
  \text{subject to} \: & \: \sum_{i=1}^N P_i u_i(k) \le P^\text{max}, \hspace{1cm} k \in \{0, \ldots, T-1\}
  \\
  & \: (u_i, e_i) \in X_i, \hspace{2.4cm} \: i \in \{1, \ldots, N\}.

The problem is recognized to be a :ref:`constraint-coupled <tutorial>` problem,
with local variables :math:`x_i` equal to the stack of :math:`e_i(k), u_i(k)` for :math:`k \in \{0, \ldots, T-1\}`,
plus :math:`e_i(T)`. The local objective function is equal to

.. math::

  f_i(x_i) = \sum_{k=0}^{T-1} P_i u_i(k) C^\text{u}(k),

the local constraint set is equal to

.. math::

  X_i = \{(e_i, u_i) \in \mathbb{R}^{T+1} \times \mathbb{R}^T \text{ such that local dynamics is satisfied} \}

and the local coupling constraint function :math:`g_i : \mathbb{R}^{2T+1} \rightarrow \mathbb{R}^T` has components

.. math::

  g_{i,k}(x_i) = P_i u_i(k) - \frac{P^\text{max}}{N}, \hspace{1cm} k \in \{0, \ldots, T-1\}.

The goal is to make each agent compute its portion :math:`x_i^\star = (e_i^\star, u_i^\star)`
of an optimal solution :math:`(x_1^\star, \ldots, x_N^\star)` of the optimization problem,
so that all of them can know their own assignment of the optimal charging schedule, given by
:math:`(u_i^\star(0), \ldots, u_i^\star(T-1))`.


### Data generation model

The data are generated according to table in [VuEs16]_ (see Appendix).

.. Simulation results
.. --------------------------------------

.. We run a comparative study with :math:`N = 50` agents with the following distributed algorithms:

.. * :ref:`Distributed Dual Subgradient <alg_dual_subgradient>`
.. * :ref:`Distributed Primal Decomposition <alg_primal_decomp>`

.. For the Distributed Primal Decomposition algorithm, we choose a sufficiently large parameter :math:`M = 100`.
.. As for the step-size, we use for both algorithms the diminishing rule :math:`\alpha^k = \frac{1}{k^{0.6}}`.

.. In the following figures we show the evolution of the two algorithms...... TODO figures:

.. * cost convergence
.. * coupling constraint value

### Complete code
--------------------------------------
.. _pev_code:

.. literalinclude:: ../../../../examples/setups/pev/launcher.py
  :caption: examples/setups/pev/launcher.py

.. literalinclude:: ../../../../examples/setups/pev/results.py
  :caption: examples/setups/pev/results.py

The two files can be executed by issuing the following commands in the example folder:

.. code-block:: bash

  > mpirun -np 50 --oversubscribe python launcher.py
  > python results.py
