# Auto-CEn: AutoML for Classifier Ensembles - Diversity-based Classifier Selection and Decision Fusion Optimization 

This repository provides the prototypical implementation of our framework Auto-CEn.
Auto-CEn is an AutoML framework to create both diversity and performance optimized ensembles.
To achieve optimized ensembles, we optimize the classifiers of the ensemble by solving the common CASH problem and optimize the decision fusion methods of the ensemble.
Furthermore, the classifiers for the ensemble are selected according to their diversity and performance to maximize the generalization performance.
For this, we use a post hoc selection method that clusters the classifiers according to their diversity.
Details can be taken from our paper "Auto-CEn: AutoML for Classifier Ensembles - Diversity-based Classifier Selection and Decision Fusion Optimization".

## Overview
The repository contains several modules and folders for reproducing the results of the paper.
The repository structure is explained in the following. Each module contains a README detailing its contents.

```md
├── evaluation: Folder containing the code for reproducing the evaluation results, as well as the results themselves.
│   ├── evaluation_results: Folder containing the results of the evaluation runs.
│   ├── benchmark_scripts: Folder containing the scripts for running the evaluation.
├── auto_cen: Contains the framework code.
│   ├── inout: Package containing modules for reading and writing data, as well as generating evaluation results.
│   ├── main: Package containing the core modules for running the framework, i.e. storing the data, constructing the configuration space and running the ensemble optimization.
│   ├── optimization: Package containing the modules for executing the CASH optimization and the data structure for storing algorithm configurations
│   ├── pipeline: Package containing the modules necessary for creating the automl and ensemble pipeline, i.e. the classification and decision fusion algorithms, the post hoc selection and the evaluation of the configurations.
│   ├── pusion: Library containing the decision fusion classes. The implementation is taken from the pusion paper by Wilhelm et al. [1].
│   ├── utils: Package holding helper modules for the framework.
│   ├── constants.py: Module holding constant values of the framework.
│   ├── example.py: Module containing example code for running the framework.
└────── requirements.txt: Text file holding the library dependencies.
```

[1] Yannick Wilhelm et al. 2023. PUSION - A Generic and Automated Framework for
Decision Fusion. In Proc. of the 39th International Conference on Data Engineering
(ICDE). IEEE, Anaheim, CA, USA, 3282–3295. https://doi.org/10.1109/ICDE55515.
2023.00252

## Installation
To use the framework Python 3.8 and Ubuntu >=22.04 are required.
Then you have to install the dependencies specified in 'auto_cen/requirements.txt' and are ready to use the framework.

## Example
In the following, examples for running the framework are given.

### Minimal example
As a minimal example, the iris dataset is used.
We first import the framework using `import automlopen as ao` and sklearn which we need for the iris dataset via `from sklearn import datasets`.
Then we load the dataset `X, y = datasets.load_iris(return_X_y=True, as_frame=True)` and call the framework `el = ao.EnsembleLearner(ens_size=5, budget_m=60, budget_f=60)`.
The parameters required for calling the framework are the ensemble size `ens_size`, the budget for the classifier optimization `budget_m` and the budget for the decision fusion optimization `budget_f`.
In this example, we choose an ensemble size of 5 and set both budgets to 1min, as the iris dataset is small.
Then we can run the framework using `el.fit_evaluate(X, y)`, which will generate us an optimized ensemble and evaluate it.
The resulting code is then:

	import automlopen as ao
	from sklearn import datasets

	X, y = datasets.load_iris(return_X_y=True, as_frame=True)

	el = ao.EnsembleLearner(ens_size=5, budget_m=60, budget_f=60)
	el.fit_evaluate(X, y)

Alternatively, the method `fit` can be used to optimize the ensemble, `predict` to use the fitted ensemble to make predictions, `evaluate` to evaluate the fitted ensemble and `save_ensemble` to save the fitted ensemble as a pickle file.

### Advanced example
Generally, we do need higher budgets for the optimization and are unsure what the optimal ensemble size is.
In this example, we show a few of the most important parameters that can be used to adapt the framework.

Finding the ensemble size: To search for an ensemble size, we can use the parameter `find_ensemble_size` and set it as `=True`.
Then, all ensemble sizes between 2 and `ensemble_size` are explored.
However, this requires longer runtimes, as the decision fusion optimization will be executed `ensemble_size - 1` times, resulting in a runtime of around `(ensemble_size - 1) * budget_f`.

Limiting the runtime of the configuration evaluation: In the optimization, multiple classifier and decision fusion configurations will be evaluated. To avoid wasting too much runtime on individual configurations, the parameter `cutoff_time` can be set. I.e. if we want to cancel evaluating a configuration after 10 minutes, we set `cutoff_time=600`.

Cross-validation: Using cross-validation often improves the results of the optimization. To use cross-validation the parameter `n_splits` can be set. E.g. setting `n_splits=10` will use 10-fold cross-validation to evaluate each classifier and decision fusion configuration in the optimization.

Changing the performance and diversity metrics: If the metrics used in optimizing the ensembles should be changed, the parameters `perf_metric` for the performance metric and `div_metric` for the diversity metric can be used.
The metrics provided by the framework are specified in auto_cen/constants.py.
**Warning**: The diversity metric has to be in the interval [0,1] with 0 being low diversity and 1 being high diversity for the post hoc method to work.
**Warning**: The performance metric will be used as a loss function by computing 1-metric. The metric should as such be in the interval [0,1], where 0 is low performance and 1 is high performance.

If additional metrics should be computed for the evaluation of the resulting ensemble (**NOT** used in the optimization), they can be specified in the parameters `eval_perf_metrics` for the performance metrics and `eval_div_metrics` for the diversity metrics.

Reproducible results: To make the results reproducible, the parameter `seed` can be used to set a random seed.

An example of a more advanced call can then be:

    el = ao.EnsembleLearner(ens_size = 20, 
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


The fit_evaluate method can be customized by specifying the dataset split to be used (`train_size, valid_size, test_size`), if stratification should be used (`stratify`), if parallelization should be used (`n_processes`) and where the evaluation results should be saved (`save_path`):

    el.fit_evaluate(X, y, n_processes=8, save_path=SAVE_PATH, stratify=y,
                    train_size=0.80, valid_size=0.0, test_size=0.20)


## Reproducibility
This repository provides the code necessary to reproduce the results of the paper "Auto-CEn: AutoML for Classifier Ensembles - Diversity-based Classifier Selection and Decision Fusion Optimization".

The scripts for generating and running the evaluation are located in 'evaluation/benchmark scripts'.
Here, 'generate_auto_cen_benchmark.py' and 'generate_autosklearn_benchmark.py' generate shell scripts that can be used to reproduce the evaluation.
The scripts 'run_auto_cen_benchmark.py' and 'run_auto_sklearn_benchmark.py' are then executed by the shell scripts for different datasets.

