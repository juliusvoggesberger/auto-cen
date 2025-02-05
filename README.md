# Auto-CEn: AutoML for Classifier Ensembles - Diversity-based Classifier Selection and Decision Fusion Optimization 

This repository provides the prototypical implementation of our framework Auto-CEn.
Auto-CEn is an AutoML framework to create classifier ensembles that are optimized for both their diversity and predictive performance.
Classifier ensembles consist of a set of classifiers and a decision fusion
method that combines the predictions of the classifiers.
To achieve optimized ensembles, Auto-CEn first optimizes the individual classifiers of the ensemble by solving the common CASH problem.
Secondly, our framework selects the classifiers to be used in the ensemble according to both their diversity and their performance in order to maximize the generalization performance.
For this, we use a post hoc selection method that clusters the classifiers according to their diversity.
Finally, Auto-CEn optimizes the decision fusion method of the ensemble.
Details can be found in our paper "Auto-CEn: AutoML for Classifier Ensembles - Diversity-based Classifier Selection and Decision Fusion Optimization".

## Overview
The repository contains several modules and folders for reproducing the results of the paper.
The repository structure is explained in the following. Each module contains a README detailing its contents.

```md
├── evaluation: Folder containing the code for reproducing the evaluation results.
│   ├── evaluation_results: Folder containing the results of the evaluation runs.
│   ├── benchmark_scripts: Folder containing the scripts for running the evaluation.
├── auto_cen: Contains the framework code.
│   ├── inout: Package containing modules for reading and writing data, as well as generating evaluation results.
│   ├── main: Package containing the core modules for running the framework, i.e. storing the data, constructing the configuration space and running the ensemble optimization.
│   ├── optimization: Package containing the modules for executing the CASH optimization and the data structure for storing algorithm configurations
│   ├── pipeline: Package containing the modules necessary for creating the AutoML and ensemble pipeline, i.e. the classification and decision fusion algorithms, the post hoc selection of classifiers, and the evaluation of the configurations.
│   ├── pusion: Library containing the decision fusion classes. The implementation is taken from the paper by Wilhelm et al. [1].
│   ├── utils: Package containing helper modules for the framework.
│   ├── constants.py: Module containing constant values of the framework.
│   ├── example.py: Module containing example code for running the framework.
└────── requirements.txt: Text file containing the library dependencies.
```

[1] Yannick Wilhelm et al. 2023. PUSION - A Generic and Automated Framework for
Decision Fusion. In Proc. of the 39th International Conference on Data Engineering
(ICDE). IEEE, Anaheim, CA, USA, 3282–3295. https://doi.org/10.1109/ICDE55515.
2023.00252

## Installation
To use the framework Python 3.8 and Ubuntu >=22.04 are required.
Then you may install the dependencies specified in 'auto_cen/requirements.txt' and are ready to use the framework.

## Example
In the following, examples for running the framework are given.

### Minimal example
As a minimal example, the [Iris](https://archive.ics.uci.edu/dataset/53/iris) dataset is used.
We first import the framework using `import auto_cen as ac`.
Furthermore, we import sklearn, which we need for the Iris dataset, via `from sklearn import datasets`.
Then we load the dataset `X, y = datasets.load_iris(return_X_y=True, as_frame=True)` and call the Auto-CEn framework `el = ao.EnsembleLearner(ens_size=5, budget_m=60, budget_f=60)`.
The parameters required for calling the framework are the ensemble size `ens_size`, the budget for the classifier optimization `budget_m` and the budget for the decision fusion optimization `budget_f`.
In this example, we choose an ensemble size of 5 and set both budgets to 1min, as the Iris dataset is rather small.
Then we can run the framework using `el.fit_evaluate(X, y)`, which will generate us an optimized ensemble.
The resulting code is then:

	import auto_cen as ac
	from sklearn import datasets

	X, y = datasets.load_iris(return_X_y=True, as_frame=True)

	el = ac.EnsembleLearner(ens_size=5, budget_m=60, budget_f=60)
	el.fit_evaluate(X, y)

Alternatively, the method `fit` can be used to optimize the ensemble, `predict` to use the fitted ensemble to make predictions, `evaluate` to evaluate the fitted ensemble and `save_ensemble` to save the fitted ensemble as a pickle file.

### Advanced usage
In this example, we show a few of the most important parameters that can be used to adapt the framework to other instances.
For instance, we may need higher budgets for the optimization or another ensemble size.

Finding the ensemble size: To search for an ensemble size, we can use the parameter `find_ensemble_size` and set it as `=True`.
Then, all ensemble sizes between 2 and `ensemble_size` are explored.
However, this requires longer runtimes, as the decision fusion optimization will be executed `ensemble_size - 1` times, resulting in a runtime of around `(ensemble_size - 1) * budget_f`.

Limiting the runtime of the evaluation of individual configurations: 
In the optimization, multiple classifier configurations and decision fusion configurations are evaluated. 
To avoid wasting too much runtime on individual configurations, the parameter `cutoff_time` can be set. 
For instance, if we want to stop the evaluation of a configuration after 10 minutes, we set `cutoff_time=600`.

Cross-validation: Using cross-validation often improves the results of the optimization. 
To use cross-validation the parameter `n_splits` can be set. 
Setting `n_splits=10` will for instance use 10-fold cross-validation to evaluate each classifier and decision fusion configuration in the optimization.

Changing the performance and diversity metrics: 
If the metrics used for optimizing the ensembles shall be changed, the parameters `perf_metric` for the performance metric and `div_metric` for the diversity metric can be used.
The metrics provided by the framework are specified in auto_cen/constants.py.
**Warning**: The diversity metric has to be in the interval [0,1] with 0 being low diversity and 1 being high diversity for the post hoc method to work.
**Warning**: The performance metric is used in a loss function by computing 1-`perf_metric`. 
The metric should hence be in the interval [0,1], where 0 is low performance and 1 is high performance.

If additional metrics are to be computed for the evaluation of the resulting ensembles (i.e., metrics that are **NOT** directly used in the optimization), they can be specified in the parameters `eval_perf_metrics` for the performance metrics and `eval_div_metrics` for the diversity metrics.

Reproducible results: To make the results reproducible, the parameter `seed` can be used to set a random seed.

An example of a more advanced call can then be:

    el = ac.EnsembleLearner(ens_size = 20, 
                            model_budget = 2400, 
                            fusion_budget = 1200,
                            find_ensemble_size=True,
                            cutoff_time= 600,
                            n_splits= 10,
                            perf_metric=BALANCED_ACCURACY,
                            div_metric=DOUBLEFAULT_NORM,
                            eval_perf_metrics=[BALANCED_ACCURACY, ACCURACY, F1_MACRO],
                            eval_div_metrics=[DOUBLEFAULT, DISAGREEMENT],
                            seed=123)


The `fit_evaluate` method can be customized by specifying the dataset split to be used (`train_size, valid_size, test_size`).
Furthermore, parameters for using stratification (`stratify`) and parallelization (`n_processes`) are provided.
The folder for saving the evaluation results is set via the `save_path` parameter.

    el.fit_evaluate(X, y, n_processes=8, save_path=SAVE_PATH, stratify=y,
                    train_size=0.80, valid_size=0.0, test_size=0.20)


## Reproducibility
This repository provides the code necessary to reproduce the results of the paper "Auto-CEn: AutoML for Classifier Ensembles - Diversity-based Classifier Selection and Decision Fusion Optimization".

The scripts for generating and running the evaluation are located in `evaluation/benchmark scripts`.
Here, `generate_auto_cen_benchmark.py`, `generate_autosklearn_benchmark.py` and `generate_divbo_benchmark.py` generate shell scripts that can be used to reproduce the evaluation.

The shell script generated by `generate_auto_cen_benchmark.py` is used to compute the evaluation results of Auto-CEn and the best individual optimized classifier baseline for each dataset.
Here, the shell script calls `run_auto_cen_benchmark.py` for each dataset, which runs Auto-CEn for each given dataset.

The shell script generated by `generate_autosklearn_benchmark.py` is used to compute the evaluation results of the auto-sklearn baseline.
Here, the shell script calls `run_auto_sklearn_benchmark.py` for each dataset, which runs auto-sklearn for each given dataset.

The shell script generated by `generate_divbo_benchmark.py` is used to compute the evaluation results of the DivBO baseline.
Here, the shell script calls `run_divbo_benchmark.py` for each dataset, which runs DivBO for each given dataset.

For QDO-ES no shell script needs to be generated.
Instead, the python script `run_qdo_es_benchmark.py` can be run directly.
