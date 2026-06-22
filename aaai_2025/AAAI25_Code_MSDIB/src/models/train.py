import torch.utils.data
import wandb
import torch as th

import numpy as np
import random
import config
import helpers
from data.load import load_dataset
from models import callback
from models.build_model import build_model
from models import evaluate
from config.defaults import Loss
import time
from models.evaluate import calc_metrics


def metrics_loss_acc_nmi(net, eval_data, batch_size):
    losses = []
    predictions = []
    labels = []
    net.eval()
    for i, (*batch, label, _) in enumerate(eval_data):
        if label.size(0) == batch_size:

            pred, _, _ = net(batch)

            predictions.append(helpers.npy(pred).argmax(axis=1))
            labels.append(helpers.npy(label))


            batch_losses = net.calc_losses(ignore_in_total='')

            losses.append(helpers.npy(batch_losses))

    net.train()
    labels = np.concatenate(labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    metrics = calc_metrics(labels, predictions)
    acc = metrics['acc']
    nmi = metrics['nmi']
    losses = helpers.dict_means(losses)

    return losses['tot'], acc, nmi


def list_txt(path, list=None):

    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist


def train(cfg, net, loader, run, eval_data, callbacks=tuple()):
    """
    Train the model for one run.

    :param cfg: Experiment config
    :type cfg: config.defaults.Experiment
    :param net: Model
    :type net:
    :param loader: DataLoder for training data
    :type loader:  th.utils.data.DataLoader
    :param eval_data: DataLoder for evaluation data
    :type eval_data:  th.utils.data.DataLoader
    :param callbacks: Training callbacks.
    :type callbacks: List
    :return: None
    :rtype: None
    """
    n_batches = len(loader)
    Loss_list = []
    Accuracy_list = []
    NMI_list = []
    time_list = []
    begin = time.time()
    max_acc = 0
    for e in range(1, cfg.n_epochs + 1):
        iter_losses = []
        for i, data in enumerate(loader):
            *batch, _, index = data

            try:
                batch_losses = net.train_step(batch, epoch=(e-1), it=i, n_batches=n_batches)
            except Exception as e:
                print(f"Training stopped due to exception: {e}")
                return

            iter_losses.append(helpers.npy(batch_losses))
        logs = evaluate.get_logs(cfg, net, eval_data=eval_data, iter_losses=iter_losses, epoch=e, include_params=True)
        if (e is None) or ((e % cfg.eval_interval) == 0):
            loss, acc, nmi = metrics_loss_acc_nmi(net, eval_data, batch_size=100)
            Loss_list.append(loss)
            Accuracy_list.append(acc)
            NMI_list.append(nmi)
            if acc > max_acc:
                max_acc = acc

        try:
            for cb in callbacks:
                cb.epoch_end(e, logs=logs, net=net)
        except callback.StopTraining as err:
            print(err)
            break

        end = time.time()
        time_list.append(end - begin)


    list_txt(path='../../acc_nmi_loss/iapr_loss.txt', list=Loss_list)
    list_txt(path='../../time_epoch/iapr_time.txt', list=time_list)
    list_txt(path='../../acc_nmi_loss/iapr_acc.txt', list=Accuracy_list)
    list_txt(path='../../acc_nmi_loss/iapr_nmi.txt', list=NMI_list)


def main():
    """
    Run an experiment.
    """
    experiment_name, cfg = config.get_experiment_config()
    dataset = load_dataset(**cfg.dataset_config.dict(), n_views=cfg.model_config.fusion_config.n_views)

    loader = th.utils.data.DataLoader(dataset, batch_size=int(cfg.batch_size), shuffle=True, num_workers=0,
                                      drop_last=True, pin_memory=False)

    eval_data = evaluate.get_eval_data(dataset, cfg.n_eval_samples, cfg.batch_size)
    experiment_identifier = wandb.util.generate_id()
    print(experiment_identifier)

    run_logs = []
    for run in range(cfg.n_runs):
        net = build_model(cfg.model_config)
        callbacks = (
            callback.Printer(print_confusion_matrix=(cfg.model_config.cm_config.n_clusters <= 100)),
            callback.ModelSaver(cfg=cfg, experiment_name=experiment_name, identifier=experiment_identifier,
                                run=run, epoch_interval=1, best_loss_term=cfg.best_loss_term,
                                checkpoint_interval=cfg.checkpoint_interval),
            callback.EarlyStopping(patience=cfg.patience, best_loss_term=cfg.best_loss_term, epoch_interval=1)
        )

        train(cfg, net, loader, run, eval_data=eval_data, callbacks=callbacks)

        run_logs.append(evaluate.eval_run(cfg=cfg, cfg_name=experiment_name,
                                          experiment_identifier=experiment_identifier, run=run, net=net,
                                          eval_data=eval_data, callbacks=callbacks, load_best=True))





if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    print(seed)
    main()

