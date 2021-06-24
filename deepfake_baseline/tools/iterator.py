# Copyright © 2020 Jungwon Choi <jungwon.choi@kaist.ac.kr>
import torch
import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from tools.utils import mean_accuracy
from tools.utils import get_criterion, get_optimizer, get_scheduler
from tools.writer import Checkpointer

class Iterator():
    #===========================================================================
    def __init__(self, args, model, dataloaders, criterion=None, optimizer=None, scheduler=None, checkpointer=None):
        self.args = args
        self.model = model
        self.device = args.device
        self.dataloaders = dataloaders
        if not args.test_only:
            self.criterion = get_criterion(args) if criterion is None else criterion
            self.optimizer = get_optimizer(args, model) if optimizer is None else optimizer
            self.scheduler = get_scheduler(args, self.optimizer) if scheduler is None else scheduler
            self.checkpointer = Checkpointer(args, model, self.optimizer) if checkpointer is None else checkpointer
        self.softmax = torch.nn.Softmax(dim=1)

        if 'mobilenet_v3' in self.args.model:
            ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: 0.0001 * averaged_model_parameter + 0.9999 * model_parameter
            self.ema_model = torch.optim.swa_utils.AveragedModel(self.model, avg_fn=ema_avg)

    #===========================================================================
    def train(self):
        self.model.train()
        phase = 'train'
        self.checkpointer.writer.count_up('current_epoch')

        # Initialize result dictionaries
        losses = []
        envs_preds, envs_labels = OrderedDict(), OrderedDict()
        for env_name in self.dataloaders[phase].keys():
            envs_preds[env_name], envs_labels[env_name] = list(), list()

        total_steps = min(list(map(len, list(self.dataloaders[phase].values()))))
        for step, envs_batch in enumerate(zip(*self.dataloaders[phase].values())):
            envs_losses = list()
            for env_name, (images, labels) in zip(self.dataloaders[phase].keys(), envs_batch):
                images = images.to(self.device)
                labels = labels.squeeze(dim=1).to(self.device)

                logits = self.model(images)

                # Calculate ERM loss
                # loss = self.criterion['ERM'](logits.squeeze(), labels)
                loss = self.criterion(logits.squeeze(), labels)
                envs_losses.append(loss)

                # Accumulate predictions and labels
                preds = self.softmax(logits)
                envs_preds[env_name].extend(preds[:,-1].cpu().detach().numpy())
                envs_labels[env_name].extend(labels.cpu().detach().numpy())

            # Total loss
            total_loss = 0.
            erm_loss = torch.stack(envs_losses).mean()
            total_loss += erm_loss
            losses.append(total_loss.item())

            # Update parameters
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            if 'mobilenet_v3' in self.args.model:
                self.ema_model.update_parameters(self.model)

            if self.scheduler is not None:
                self.scheduler.step()

            self.checkpointer.writer.count_up('current_iter')
            # self.checkpointer.writer.write_result('train_iteration_loss', total_loss.item(), self.checkpointer.writer.current_iter)

            # Print the current process
            if self.checkpointer.writer.current_iter % self.args.print_freq == 0:
                avg_loss = sum(losses)/len(losses)
                print('\r(Train) Step: [{:05d}/{:05d}] Iter: {:5d} --> cur.loss {:.1E} / avg.loss: {:.1E}'.format(step+1, total_steps,
                                                                                                                    self.checkpointer.writer.current_iter,
                                                                                                                    total_loss.item(), avg_loss), end='')
            # Validate model
            if self.args.num_gpus > 0 and (self.args.local_rank % self.args.num_gpus == 0):
                if self.checkpointer.writer.current_iter % self.args.val_freq == 0:
                    self.validate()
                    self.model.train()

            # Stop training if max_iters reached
            if self.args.max_iters is not None and self.checkpointer.writer.current_iter >= self.args.max_iters:
                print('\n[Notice] The maximun iteration has been reached..')
                self.checkpointer.stop_training = True
                break

        # if self.scheduler is not None:
        #     self.scheduler.step()

        avg_loss = sum(losses)/len(losses)
        self.checkpointer.writer.write_result('train_loss', avg_loss, self.checkpointer.writer.current_epoch)
        self.save_results(envs_preds, envs_labels, phase='train', avg_loss=avg_loss)

    #===========================================================================
    def validate(self):
        self.model.eval()
        phase = 'val'
        self.checkpointer.writer.write_result('val_iter', self.checkpointer.writer.current_iter)

        # Initialize result dictionaries
        losses = []
        envs_preds, envs_labels = OrderedDict(), OrderedDict()
        for env_name in self.dataloaders[phase].keys():
            envs_preds[env_name], envs_labels[env_name] = list(), list()

        total_steps = min(list(map(len, list(self.dataloaders[phase].values()))))
        with torch.no_grad():
            for env_name, dataloader in zip(self.dataloaders[phase].keys(), self.dataloaders[phase].values()):
                total_steps = len(dataloader)
                envs_losses = list()
                for step, (images, labels) in enumerate(dataloader):
                    images = images.to(self.device)
                    labels = labels.squeeze(dim=1).to(self.device)

                    logits = self.model(images)

                    # Calculate ERM loss
                    # loss = criterion['ERM'](logits.squeeze(), labels)
                    loss = self.criterion(logits.squeeze(), labels)
                    envs_losses.append(loss)

                    # Accumulate predictions and labels
                    preds = self.softmax(logits)
                    envs_preds[env_name].extend(preds[:,-1].cpu().detach().numpy())
                    envs_labels[env_name].extend(labels.cpu().detach().numpy())
                    print('\r(Val) Step: [{:05d}/{:05d}] {:<20}'.format(step+1, total_steps, env_name), end='')
                # Total loss
                total_loss = 0.
                erm_loss = torch.stack(envs_losses).mean()
                total_loss += erm_loss
                losses.append(total_loss.item())

        avg_loss = sum(losses)/len(losses)
        self.checkpointer.writer.write_result('val_loss', avg_loss, self.checkpointer.writer.current_iter)
        self.save_results(envs_preds, envs_labels, phase='val', avg_loss=avg_loss)



    #===========================================================================
    def test(self):
        self.model.eval()
        phase = 'test'

        envs_preds, envs_labels = OrderedDict(), OrderedDict()
        for env_name in self.dataloaders[phase].keys():
            envs_preds[env_name], envs_labels[env_name] = list(), list()

        total_steps = min(list(map(len, list(self.dataloaders[phase].values()))))
        with torch.no_grad():
            for env_name, dataloader in zip(self.dataloaders[phase].keys(), self.dataloaders[phase].values()):
                total_steps = len(dataloader)
                for step, (images, labels) in enumerate(dataloader):
                    images = images.to(self.device)
                    labels = labels.squeeze(dim=1)

                    logits = self.model(images)

                    # Accumulate predictions and labels
                    preds = self.softmax(logits)
                    envs_preds[env_name].extend(preds[:,-1].cpu().detach().numpy())
                    envs_labels[env_name].extend(labels.numpy())
                    print('\r(Test) Step: [{:05d}/{:05d}] {:<20}'.format(step+1, total_steps, env_name), end='')

        self.save_results(envs_preds, envs_labels, phase='test')

    #===========================================================================
    def save_results(self, envs_preds, envs_labels, phase, avg_loss=None):
        results = OrderedDict()
        total_preds = list()
        total_labels = list()
        total_unseen_preds = list()
        total_unseen_labels = list()

        train_domains = self.args.domain_list

        # Calcualte each environment's socres
        for env_name in envs_labels.keys():
            e_preds = torch.Tensor(envs_preds[env_name])
            e_labels = torch.Tensor(envs_labels[env_name])
            avg_acc = mean_accuracy(e_preds, e_labels)
            real_acc = mean_accuracy(e_preds[e_labels==0], e_labels[e_labels==0])
            fake_acc = mean_accuracy(e_preds[e_labels==1], e_labels[e_labels==1])
            roc_auc = roc_auc_score(np.array(envs_labels[env_name]), np.array(envs_preds[env_name]))

            if not self.args.test_only and self.checkpointer is not None:
                self.checkpointer.writer.write_result('{}_iteration_acc_{}'.format(phase, env_name), avg_acc, self.checkpointer.writer.current_iter, tb_only=True)

            results[env_name] = {'avg_acc': avg_acc, 'real_acc': real_acc, 'fake_acc': fake_acc, 'roc_auc': roc_auc,
                                 'num_real': len(e_labels[e_labels==0]), 'num_fake': len(e_labels[e_labels==1])}

            if env_name in train_domains:
                total_preds.extend(envs_preds[env_name])
                total_labels.extend(envs_labels[env_name])
            if env_name not in train_domains:
                total_unseen_preds.extend(envs_preds[env_name])
                total_unseen_labels.extend(envs_labels[env_name])

        # Calculate Acc. and ROC AUC score for all (seen) data
        if len(total_preds)>0:
            t_preds = torch.Tensor(total_preds)
            t_labels = torch.Tensor(total_labels)
            total_avg_acc = mean_accuracy(t_preds, t_labels)
            total_real_acc = mean_accuracy(t_preds[t_labels==0], t_labels[t_labels==0])
            total_fake_acc = mean_accuracy(t_preds[t_labels==1], t_labels[t_labels==1])
            total_roc_auc = roc_auc_score(np.array(t_labels), np.array(t_preds))
            if not self.args.test_only and self.checkpointer is not None:
                self.checkpointer.writer.write_result('{}_acc'.format(phase), total_avg_acc, self.checkpointer.writer.current_iter, tb_only=(phase=='test'))
                self.checkpointer.writer.write_result('{}_roc_auc'.format(phase), total_roc_auc, self.checkpointer.writer.current_iter, tb_only=(phase=='test'))
                self.checkpointer.writer.write_result('{}_iteration_acc'.format(phase), total_avg_acc, self.checkpointer.writer.current_iter, tb_only=True)
                self.checkpointer.writer.write_result('{}_iteration_roc_auc'.format(phase), total_roc_auc, self.checkpointer.writer.current_iter, tb_only=True)
            results['Total (seen)'] = {'avg_acc': total_avg_acc, 'real_acc': total_real_acc, 'fake_acc': total_fake_acc,
                                'num_real': len(t_labels[t_labels==0]), 'num_fake': len(t_labels[t_labels==1]), 'roc_auc': total_roc_auc}
        else:
            if not self.args.test_only and self.checkpointer is not None:
                self.checkpointer.writer.write_result('{}_acc'.format(phase), 0, self.checkpointer.writer.current_iter, tb_only=(phase=='test'))
                self.checkpointer.writer.write_result('{}_roc_auc'.format(phase), 0, self.checkpointer.writer.current_iter, tb_only=(phase=='test'))

        # Calculate Acc. and ROC AUC score for all (unseen) data
        if phase!='train':
            if len(total_unseen_preds)>0:
                t_preds = torch.Tensor(total_unseen_preds)
                t_labels = torch.Tensor(total_unseen_labels)
                total_avg_acc = mean_accuracy(t_preds, t_labels)
                total_real_acc = mean_accuracy(t_preds[t_labels==0], t_labels[t_labels==0])
                total_fake_acc = mean_accuracy(t_preds[t_labels==1], t_labels[t_labels==1])
                total_roc_auc = roc_auc_score(np.array(t_labels), np.array(t_preds))
                if not self.args.test_only and self.checkpointer is not None:
                    if phase=='val':
                        self.checkpointer.writer.write_result('val_acc_us', total_avg_acc, self.checkpointer.writer.current_iter)
                        self.checkpointer.writer.write_result('val_roc_auc_us', total_roc_auc, self.checkpointer.writer.current_iter)
                    self.checkpointer.writer.write_result('{}_iteration_acc_us'.format(phase), total_avg_acc, self.checkpointer.writer.current_iter, tb_only=True)
                    self.checkpointer.writer.write_result('{}_iteration_roc_auc_us'.format(phase), total_roc_auc, self.checkpointer.writer.current_iter, tb_only=True)
                results['Total (unseen)'] = {'avg_acc': total_avg_acc, 'real_acc': total_real_acc, 'fake_acc': total_fake_acc,
                                    'num_real': len(t_labels[t_labels==0]), 'num_fake': len(t_labels[t_labels==1]), 'roc_auc': total_roc_auc}
            else:
                if phase=='val':
                    self.checkpointer.writer.write_result('val_acc_us', 0, self.checkpointer.writer.current_iter)
                    self.checkpointer.writer.write_result('val_roc_auc_us', 0, self.checkpointer.writer.current_iter)

        # Print results
        if phase != 'test' and self.args.dataset is None:
            print(' -----> {} avg.loss: {:.3E} / acc.: {:6.2f} (R {:6.2f} / F {:6.2f}) / roc auc score: {:.4f}'.format(phase, avg_loss, total_avg_acc*100, total_real_acc*100, total_fake_acc*100, total_roc_auc))
        if phase != 'train':
            print(' -----> environment results')
            for env_name, result in results.items():
                if 'Total' not in env_name:
                    if env_name in train_domains:
                        print(' {:<15}{:<10}acc.: {:6.2f} (R {:6.2f} / F {:6.2f}){:2}[{:4d}/{:4d}] / roc auc score: {:.4f}'.format(env_name, '(seen)', result['avg_acc']*100,
                                result['real_acc']*100, result['fake_acc']*100, '', result['num_real'], result['num_fake'], result['roc_auc']))
                    else:
                        print(' {:<15}{:<10}acc.: {:6.2f} (R {:6.2f} / F {:6.2f}){:2}[{:4d}/{:4d}] / roc auc score: {:.4f}'.format(env_name, '(unseen)', result['avg_acc']*100,
                                result['real_acc']*100, result['fake_acc']*100, '', result['num_real'], result['num_fake'], result['roc_auc']))
                else:
                    if '(seen)' in env_name:
                        print(' {:<15}{:<10}acc.: {:6.2f} (R {:6.2f} / F {:6.2f}){:2}[{:4d}/{:4d}] / roc auc score: {:.4f}'.format(env_name, '', result['avg_acc']*100,
                            result['real_acc']*100, result['fake_acc']*100, '', result['num_real'], result['num_fake'], result['roc_auc']))
                    if 'unseen' in env_name:
                        print(' {:<15}{:<10}acc.: {:6.2f} (R {:6.2f} / F {:6.2f}){:2}[{:4d}/{:4d}] / roc auc score: {:.4f}'.format(env_name, '', result['avg_acc']*100,
                            result['real_acc']*100, result['fake_acc']*100, '', result['num_real'], result['num_fake'], result['roc_auc']))

        if phase=='val':
            score = results['Total (seen)']['roc_auc']
            self.checkpointer.save_checkpoint(score=score, verbose=self.args.verbose)
