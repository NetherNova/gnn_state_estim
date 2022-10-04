__author__ = 'martin.ringsquandl'

import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import ParameterGrid
from torch_geometric.data import DataLoader

from src.dataset.example_nets import ExampleNets
from src.learning.experiments import *
from src.dataset.utils import normalize

MAX_PATIENCE = 3


class Trainer(object):
    def __init__(self):
        self.loss_fn = torch.nn.MSELoss()
        # TODO try:
        # self.loss_fn = torch.nn.SmoothL1Loss()

    def train(self, model, device, lr, loader, epochs, min_ys, max_ys, exp_targets=False, task_weights=[0.3, 0.3, 0.3, 0.1]):
        self.patience = 0
        num_tasks = len(task_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-8)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

        self.min_ys = min_ys
        self.max_ys = max_ys

        self.scale = self.max_ys - self.min_ys
        if exp_targets:
            self.target_func = np.exp
        else:
            self.target_func = lambda x: x

        train = []
        val = []
        test = []

        masked_out_tasks = np.where(np.array(task_weights) == 0)[0]

        model.train()
        for epoch in range(1, epochs):
            # for unequal batch size correction store num observations
            losses_per_task = [0] * num_tasks
            scaled_losses_per_task = [0] * num_tasks
            train_observations_per_task = [0] * num_tasks

            val_loss_per_task = [0] * num_tasks
            scaled_val_loss_per_task = [0] * num_tasks
            val_observations_per_task = [0] * num_tasks

            test_loss_per_task = [0] * num_tasks
            scaled_test_loss_per_task = [0] * num_tasks
            test_observations_per_task = [0] * num_tasks
            # batches of graphs
            for k, data in enumerate(loader):
                second_dim_mask = torch.tensor(np.array(1 - np.isnan(data.y), dtype=np.bool))
                optimizer.zero_grad()
                out = model(data)
                out = torch.squeeze(out)

                first_mask = data.train_mask
                # no labels in this batch -> skip
                if first_mask.sum() == 0:
                    continue
                second_mask = second_dim_mask[first_mask]

                loss = 0

                for task in range(num_tasks):
                    if second_mask[:, task].sum() == 0:
                        continue

                    task_loss = task_weights[task] * self.loss_fn(out[first_mask, task][second_mask[:, task]],
                                                              data.y[first_mask, task][second_mask[:, task]])
                    loss += task_loss

                    scaled_out = self.target_func(out[first_mask, task][second_mask[:, task]].detach()) * self.scale[task] + self.min_ys[task]
                    scaled_labels = self.target_func(data.y[first_mask, task][second_mask[:, task]]) * self.scale[task] + self.min_ys[task]
                    scaled_losses_per_task[task] += np.abs(scaled_out.detach() - scaled_labels).sum()

                    num_observations = second_mask[:, task].sum().item()
                    losses_per_task[task] += task_loss.item() * num_observations     # undo mean per task
                    train_observations_per_task[task] += num_observations

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    model.eval()
                    out = model(data)
                    out = torch.squeeze(out)

                    # validation
                    first_mask = data.val_mask
                    second_mask = second_dim_mask[first_mask]

                    for task in range(num_tasks):
                        if second_mask[:, task].sum() == 0:
                            continue
                        # do not report validation loss on masked out tasks
                        if task in masked_out_tasks:
                            continue

                        val_out = self.target_func(out[first_mask, task][second_mask[:, task]])
                        scaled_val_out = val_out * self.scale[task] + self.min_ys[task]

                        val_labels = self.target_func(data.y[first_mask, task][second_mask[:, task]])
                        scaled_val_labels = val_labels * self.scale[task] + self.min_ys[task]

                        tmp_loss = np.abs(scaled_val_out - scaled_val_labels)
                        val_loss = tmp_loss.sum()

                        num_observations = second_mask[:, task].sum().item()
                        scaled_val_loss_per_task[task] += val_loss.item()
                        val_observations_per_task[task] += num_observations

                    # test
                    first_mask = data.test_mask
                    second_mask = second_dim_mask[first_mask]

                    for task in range(num_tasks):
                        if second_mask[:, task].sum() == 0:
                            continue
                        # do not report validation loss on masked out tasks
                        if task in masked_out_tasks:
                            continue

                        test_out = self.target_func(out[first_mask, task][second_mask[:, task]])
                        scaled_test_out = test_out * self.scale[task] + self.min_ys[task]

                        test_labels = self.target_func(data.y[first_mask, task][second_mask[:, task]])
                        scaled_test_labels = test_labels * self.scale[task] + self.min_ys[task]

                        tmp_loss = np.abs(scaled_test_out - scaled_test_labels)
                        test_loss = tmp_loss.sum()

                        num_observations = second_mask[:, task].sum().item()
                        scaled_test_loss_per_task[task] += test_loss.item()
                        test_observations_per_task[task] += num_observations

            # scheduler.step()
            train.append(np.array(losses_per_task) / np.array(train_observations_per_task))
            val.append(np.array(scaled_val_loss_per_task) / np.array(val_observations_per_task))
            test.append(np.array(scaled_test_loss_per_task) / np.array(test_observations_per_task))

            # print every epoch
            if len(val) > 1:
                # is the sum of all (enabled) tasks
                if np.nansum(val[-1]) > np.nansum(val[-2]):
                    self.patience += 1
                else:
                    self.patience = 0

            if self.patience == MAX_PATIENCE:
                print("Early stopping on validation loss")
                return np.nansum(train[-1]), np.nansum(val[-1]), np.nansum(test[-1])

            if epoch % 1 == 0:
                train_format = ' '.join(['{:.4f}']*num_tasks)
                val_format = ' '.join(['{:.4f}']*num_tasks)
                test_format = ' '.join(['{:.4f}']*num_tasks)
                output_string = "Epoch {} - " + "Train: " + train_format + " - Val: " + val_format + " - Test: " + test_format
                print(output_string.format(epoch, *[v for v in train[-1]], *[v for v in val[-1]], *[v for v in test[-1]]))
        return np.nansum(train[-1]), np.nansum(val[-1]), np.nansum(test[-1])

    def test(self, model, loader):
        total_test_loss = 0
        total_num_nodes = 0
        for k, data in enumerate(loader):
            second_dim_mask = torch.tensor(np.array(1 - np.isnan(data.y), dtype=np.bool))
            with torch.no_grad():
                model.eval()
                out = model(data)
                out = torch.squeeze(out)

                first_mask = data.test_mask
                if first_mask.sum() == 0:
                    continue
                second_mask = second_dim_mask[first_mask]

                loss = self.loss_fn(out[first_mask][second_mask], data.y[first_mask][second_mask])

                num_nodes = out[first_mask].size()[0]
                total_test_loss += loss.item() * num_nodes
                total_num_nodes += num_nodes
        print("----")
        print("Test: ", total_test_loss / total_num_nodes)
        return total_test_loss / total_num_nodes

    def save_config(self, processed_dir, best_params, normalizers, directed):
        x_mins, x_maxs, y_mins, y_maxs = normalizers
        with open(os.path.join(processed_dir, 'encoder_dict.json'), 'r') as fp:
            encoders = json.load(fp)

        config_dict = {
            'path2dataset': './dummy/TestNetze/Subsystem_50540',
            'state_dict_file': './best_model.pt',
            'path2featurefile': './ETL_columns.csv',
            'hidden_dim': best_params['hidden_dim'],
            'heads': best_params['heads'],
            'input_dim': best_params['input_dim'],
            'directed': directed,

            'x_mins': x_mins.tolist(),
            'x_maxs': x_maxs.tolist(),
            'y_mins': y_mins.tolist(),
            'y_maxs': y_maxs.tolist(),
            'encoders': encoders,
            'targets': ['cMvI', 'cMvP', 'cMvQ', 'cMvV'],
            'server': {
                'host': "0.0.0.0",
                'port': 9999,
                'debug': False,
            },
            'output': {
                'folder_path': './output'
            },
            'app': {
                'default_input': './dummy/TestNetze/Subsystem_50540',
                'basic_checks': True,
                'advanced_checks': True
            }
        }
        with open(os.path.join(processed_dir, 'config.yaml'), 'w') as fp:
            yaml.safe_dump(config_dict, fp)

    def run_experiments(self, root_dir, network_dirs, experimenter, directed=False, augment=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = ExampleNets(root_dir, network_dirs, directed=directed, augment=augment)
        processed_dir = os.path.join(root_dir, 'processed')
        dataset, normalizers = normalize(dataset)
        df_normalizers = pd.DataFrame(normalizers, index=['x_mins', 'x_maxs', 'y_mins', 'y_maxs'])
        df_normalizers.to_csv(os.path.join(processed_dir, 'norms.csv'))

        model_params = experimenter.get_model_param_grid()
        trainer_params = experimenter.get_trainer_param_grid()
        trainer = Trainer()
        configs_model = ParameterGrid(model_params)
        configs_trainer = ParameterGrid(trainer_params)

        results = []
        best_model = None
        best_error = np.inf
        best_params = None
        i = 0
        for model_config in configs_model:
            for trainer_config in configs_trainer:
                i = i + 1
                print('Training model {}: {:d} of {:d} '.format(experimenter.model_func, i, len(configs_model) * len(configs_trainer)))

                loader = DataLoader(dataset, batch_size=trainer_config['batch_size'], shuffle=True)

                model = experimenter.model_func(num_outputs=4, **model_config).to(device)
                # model, device, lr, loader, epochs
                train_error, val_error, test_error = trainer.train(model, device,
                                                                   lr=trainer_config['lr'],
                                                                   loader=loader,
                                                                   epochs=trainer_config['epochs'],
                                                                   min_ys=normalizers[2],
                                                                   max_ys=normalizers[3],
                                                                   task_weights=trainer_config['task_weights'])
                if val_error < best_error:
                    best_error = val_error
                    best_model = model
                    # save the currently best model to disk
                    with open(os.path.join(processed_dir, 'best_model_' + str(experimenter) + '.pt'), 'wb') as fp:
                        torch.save(model.state_dict(), fp, pickle_module=pickle,
                                   pickle_protocol=pickle.HIGHEST_PROTOCOL)
                    # merge both param dicts
                    best_params = {}
                    best_params.update(model_config)
                    best_params.update(trainer_config)

                # merge both param dicts
                all_params_dict = {}
                all_params_dict.update(model_config)
                all_params_dict.update(trainer_config)
                all_params_dict['val_error'] = val_error
                all_params_dict['train_error'] = train_error
                all_params_dict['test_error'] = test_error
                results.append(all_params_dict)

                pd.DataFrame(results).sort_values('val_error').to_csv(os.path.join(processed_dir, 'results_' + str(experimenter) + '.csv'))

        test_error = trainer.test(best_model, loader)
        print("Best model test error: ", test_error)

        self.save_config(processed_dir, best_params, normalizers, directed)

        return best_model


if __name__ == '__main__':
    random_seed = 23
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    trainer = Trainer()
    ex = ExperimentTransfEnc(num_features=75)
    trainer.run_experiments('../../data/',
                            ['SimulatedPowerFlowNetworks'],
                            experimenter=ex, directed=False, augment=False)
