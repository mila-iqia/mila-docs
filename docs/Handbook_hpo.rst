Hyperparameter Optimization
===========================

Why it matters
--------------




RL paper
Are GANs created equal? a large-scale study. Lucic et al (2017)
On the state of the art of evaluation in neural language models. Melis et al (2017)
Knowledge Base Completion: Baselines Strike Back. Kadlec et al (2017)
Repro paper
Accounting for variance in ML Benchmarks
Optimizer benchmark (DeepOBS?)

Survey





Defining the search space
-------------------------

- Avoid turning discrete values into categories
- Use logarithmic distribution if the effect of hyperparameter on the objective is logarithmic (ex:
learning rates)
- Inverse logarithmic for hyperparameters behaving the other way around (ex: gamma of exponential learning rate schedule) (todo: get back equation from slides)

Learning-rate
^^^^^^^^^^^^^


Learning-rate schedule
^^^^^^^^^^^^^^^^^^^^^^


Momentum
^^^^^^^^

Weight decay
^^^^^^^^^^^^

Mini-batch size
^^^^^^^^^^^^^^^

Size of layers
^^^^^^^^^^^^^^

Number of layers
^^^^^^^^^^^^^^^^


Choosing the budget
-------------------


.. image:: _static/dim_1.png
  :width: 300
  :alt: Alternative text
  :align: center

.. image:: _static/dim_2.png
  :width: 300
  :alt: Alternative text
  :align: center

.. image:: _static/dim_3.png
  :width: 300
  :alt: Alternative text
  :align: center

.. image:: _static/easy_vs_hard.png
  :width: 600
  :alt: Alternative text
  :align: center




Selecting HPO algorithms
------------------------


.. image:: _static/handbook_hpo_algo_selection.png
  :width: 800
  :alt: Alternative text
  :align: center


How many dimensions?
^^^^^^^^^^^^^^^^^^^^


.. image:: _static/grid_vs_random.png
  :width: 500
  :alt: Alternative text
  :align: center

Computational time per trial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Late learners
^^^^^^^^^^^^^

How many trials?
^^^^^^^^^^^^^^^^

Are all dimensions continuous?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



Visualizations
--------------


.. image:: _static/regret_thumbnail.png
  :width: 500
  :alt: Alternative text
  :align: center


.. image:: _static/parallel_coordinates_select.gif
  :width: 500
  :alt: Alternative text
  :align: center


.. image:: _static/par_dep_thumbnail.png
  :width: 500
  :alt: Alternative text
  :align: center





Frameworks
----------

Oríon
^^^^^

- Developped at Mila

Ray-Tune
^^^^^^^^

- Advantage if using Ray
- Many algorithms


Optuna
^^^^^^

- Good TPE implementation
- Less algorithms




References
----------
AutoML book
