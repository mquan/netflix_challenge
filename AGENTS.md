# Experiment scope

The goal is to reach 0.85725 RMSE using predictions from models we train
ourselves on the Netflix ratings. Published methods are welcome when we train
the models and generate the predictions ourselves.

Plan and run new experiments only with those predictions. Exclude released
base-model predictions, released intermediate or final blends, and any
locally fitted model that consumes them, even indirectly (for example,
NN385). Check the inputs of a cached blend before reusing it.

The 619-column public-bank result and its ablations are historical references.
Use the fresh-training Own17x + CF96 work and its own-model successors as
baselines for new experiments.
