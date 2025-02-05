from utils.data_loader import load_train_test_data
from utils.space import get_space, get_small_space
from utils.evaluate import evaluate
from ensembles.ensemble_selection import EnsembleSelection
from utils.metric import diversity_demo
from sklearn.metrics import f1_score, roc_auc_score,  accuracy_score, balanced_accuracy_score

import argparse
import numpy as np
import pickle as pkl
from functools import partial
from mindware.components.metrics.metric import get_metric

import sys

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str)
parser.add_argument('--task_type', type=str, default='cls', choices=['cls', 'rgs'])
parser.add_argument('--rep_num', type=int, default=5)
parser.add_argument('--algos', type=str, default='rs')
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--iter_num', type=int, default=200)
parser.add_argument('--ens_size', type=int, default=25)
parser.add_argument('--beta', type=float, default=0.025)
parser.add_argument('--tau', type=float, default=0.2)

args = parser.parse_args()
datasets = args.datasets.split(',')
task_type = args.task_type
rep_num = args.rep_num
algos = args.algos.split(',')
start_id = args.start_id
iter_num = args.iter_num
ens_size = args.ens_size
alpha = args.beta
beta = args.tau

config_space = get_space(task_type=task_type)
time_limit_per_trial = 600
scorer = get_metric('bal_acc') if task_type == 'cls' else get_metric('mse')

print(f'Scorer {scorer}')
binary_cls = True

for dataset in datasets:
    for algo in algos:

        print("dataset: %s, algo: %s" % (dataset, algo))

        train_node, test_node = load_train_test_data(dataset=dataset, data_dir='../evaluation_results/', test_size=0.2,
                                                     task_type=0 if task_type == 'cls' else 4)

        eval_func = partial(evaluate,
                            scorer=scorer,
                            data_node=train_node, test_node=test_node, task_type=task_type,
                            resample_ratio=1.0, seed=1)


        for i in range(start_id, start_id + rep_num):
            csv_dic = {
                'dataset': dataset,
                'algo': algo,
                'c_budget': 3600,
                'ens_size': ens_size,
                'beta':alpha,
                'tau':beta,
                'cut_off_time': time_limit_per_trial

            }
            test_size = 0.2
            seed = 1
            if task_type == 'cls':
                from sklearn.model_selection import StratifiedShuffleSplit

                ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            else:
                from sklearn.model_selection import ShuffleSplit

                ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            for train_index, test_index in ss.split(train_node.data[0], train_node.data[1]):
                _y_train, _y_val = train_node.data[1][train_index], train_node.data[1][test_index]

            if len(np.unique(_y_train)) > 2:
                binary_cls = False

            #'''
            if algo == 'rs':
                from searchers.random_search import RandomSearch

                optimizer = RandomSearch(config_space=config_space, eval_func=eval_func, iter_num=iter_num,
                                         task_name=dataset,
                                         save_dir='./results/%s' % task_type)
            elif algo == 'bo':
                from searchers.bayesian_optimization import BayesianOptimization

                optimizer = BayesianOptimization(config_space=config_space, eval_func=eval_func, iter_num=iter_num,
                                                 task_name=dataset,
                                                 save_dir='./results/%s' % task_type,
                                                 surrogate_type='prf')
            elif algo == 'rb':
                from searchers.rising_bandit import RisingBandit

                optimizer = RisingBandit(config_space=config_space, eval_func=eval_func, iter_num=iter_num,
                                         task_name=dataset,
                                         save_dir='./results/%s' % task_type,
                                         surrogate_type='prf')
            elif algo == 'eo':
                from searchers.bayesian_optimization_ensemble import BayesianOptimizationEnsemble

                optimizer = BayesianOptimizationEnsemble(config_space=config_space, eval_func=eval_func,
                                                         iter_num=iter_num,
                                                         task_name=dataset,
                                                         save_dir='./results/%s' % task_type,
                                                         surrogate_type='prf',
                                                         scorer=scorer,
                                                         task_type=task_type,
                                                         train_node=train_node,
                                                         test_node=test_node)
            elif algo == 'bo_div':
                from searchers.bayesian_optimization_diversity import BayesianOptimizationDiversity

                optimizer = BayesianOptimizationDiversity(config_space=config_space, eval_func=eval_func,
                                                          iter_num=iter_num,
                                                          task_name=dataset,
                                                          save_dir='./results/%s' % task_type,
                                                          surrogate_type='prf',
                                                          scorer=scorer,
                                                          task_type=task_type,
                                                          ens_size=ens_size,
                                                          val_y_labels=_y_val,
                                                          alpha=alpha,
                                                          beta=beta)
            elif algo == 'rea_es':
                from searchers.rea_es import RegularizedEAEnsemble

                optimizer = RegularizedEAEnsemble(config_space=config_space, eval_func=eval_func,
                                                  iter_num=iter_num,
                                                  task_name=dataset,
                                                  save_dir='./results/%s' % task_type,
                                                  scorer=scorer,
                                                  task_type=task_type,
                                                  ens_size=ens_size,
                                                  val_y_labels=_y_val)

            optimizer.run(time_limit_per_trial=time_limit_per_trial)

            save_path = optimizer.save_path

            print(f'Path {save_path}')
            with open(save_path, 'rb') as f:
                observations = pkl.load(f)

            with open(save_path, 'wb') as f:
                pkl.dump([observations, _y_val, test_node.data[1]], f)

            with open(save_path, 'rb') as f:
                observations, val_labels, test_labels = pkl.load(f)


            val_pred_list = []
            test_pred_list = []

            if algo == 'eo':
                for ob in observations:
                    _, val_perf, test_perf, val_pred, test_pred, _ = ob
                    if val_pred is not None:
                        val_pred_list.append(val_pred)
                        test_pred_list.append(test_pred)
                        best_val = val_perf
                        best_test = test_perf
            elif algo == 'rea_es':
                best_val = np.inf
                best_test = np.inf
                for ob in observations[-20:]:
                    _, val_perf, test_perf, val_pred, test_pred, _ = ob
                    if val_pred is not None:
                        val_pred_list.append(val_pred)
                        test_pred_list.append(test_pred)
                    if val_perf < best_val:
                        best_val = val_perf
                        best_test = test_perf
            else:
                best_val = np.inf
                best_test = np.inf
                for ob in observations:
                    _, val_perf, test_perf, val_pred, test_pred, _ = ob
                    if val_pred is not None:
                        val_pred_list.append(val_pred)
                        test_pred_list.append(test_pred)
                    if val_perf < best_val:
                        best_val = val_perf
                        best_test = test_perf

            ensemble_builder = EnsembleSelection(ensemble_size=ens_size,
                                                 task_type=task_type,
                                                 scorer=scorer)
            ensemble_builder.fit(val_pred_list, val_labels)

            ens_val_pred, predictions_selected_models = ensemble_builder.predict(val_pred_list)

            if task_type == 'cls':
                ens_val_pred = np.argmax(ens_val_pred, axis=-1)

            print('Best validation perf: %s' % str(-best_val))
            print('Ensemble validation perf: %s' % str(ensemble_builder.scorer._score_func(ens_val_pred, _y_val)))

            ens_test_pred, predictions_selected_models = ensemble_builder.predict(test_pred_list)
            print(f'ens:size {ensemble_builder.ensemble_size}')



            div_dic = diversity_demo(predictions_selected_models, test_node.data[1])

            csv_dic.update(div_dic)


            if task_type == 'cls':
                ens_test_score =np.copy(ens_test_pred)
                print(f'ens_test_pred {ens_test_pred}')
                ens_test_pred = np.argmax(ens_test_pred, axis=-1)

            print('Best test perf: %s' % str(-best_test))
            print(
                'Ensemble test perf: %s' % str(ensemble_builder.scorer._score_func(ens_test_pred, test_node.data[1])))
            print(ensemble_builder.model_idx)
            if task_type == 'cls':

                csv_dic["f1_macro"] = f1_score(test_node.data[1], ens_test_pred, average="macro", labels=None)
                print(f' csv_dic["f1_macro"] { csv_dic["f1_macro"]}')
                if binary_cls:
                    csv_dic["roc_auc_ovr"] = roc_auc_score(test_node.data[1], ens_test_pred, average='weighted',
                                                           multi_class='ovr')
                    csv_dic["roc_auc_ovo"] = roc_auc_score(test_node.data[1], ens_test_pred, average='weighted',
                                                           multi_class='ovo')

                else:

                    if np.isnan(ens_test_score).any():
                        csv_dic["roc_auc_ovr"] = np.nan
                        csv_dic["roc_auc_ovo"] = np.nan

                    else:
                        csv_dic["roc_auc_ovr"] = roc_auc_score( test_node.data[1],ens_test_score,average='weighted',multi_class='ovr')
                        csv_dic["roc_auc_ovo"] = roc_auc_score(test_node.data[1], ens_test_score, average='weighted',
                                                           multi_class='ovo')

                csv_dic["accuracy"] = accuracy_score(test_node.data[1], ens_test_pred)
                csv_dic['balanced_accuracy'] = balanced_accuracy_score(test_node.data[1], ens_test_pred)
                print(
                    'Ensemble test accuracy: %s' % str(
                        csv_dic["accuracy"]))

                for key, value in csv_dic.items():
                    print(f"Key: {key}, Value: {value}")


                # ----------------- RESULT WRITER --------------------------
                import csv
                import os


                def write_dict_to_csv(file_path_csv, data_dict):
                    file_exists = os.path.isfile(file_path_csv)

                    with open(file_path_csv, mode='a', newline='') as csv_file:
                        fieldnames = data_dict.keys()
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                        if not file_exists:
                            writer.writeheader()  # Write header only if file does not already exist

                        writer.writerow(data_dict)

                file_path =  'DivBO_evaluation_runs.csv'
                write_dict_to_csv(file_path, csv_dic)

            sys.stdout.flush()
