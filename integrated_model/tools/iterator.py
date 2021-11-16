# Copyright © 2020-2021 Jungwon Choi <jungwon.choi@kaist.ac.kr>
import torch
import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, roc_curve
from tools.utils import mean_accuracy, contrastive_loss
from tools.utils import printflush as print
from tools.utils import get_criterion, get_optimizer, get_scheduler
from tools.utils import get_oudefend_output
from tools.writer import Checkpointer
from PIL import Image
from torchvision.transforms import ToPILImage
import os
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import torchvision
import matplotlib.pyplot as plt

class Iterator():
    #===========================================================================
    def __init__(self, args, model, dataloaders, criterion=None, optimizer=None, scheduler=None, checkpointer=None, best_threshold=0.5, pretrain=False):
        self.args = args
        self.model = model
        self.device = args.device
        self.dataloaders = dataloaders
        self.pretrain = pretrain # args.self_supervised
        self.self_supervised = args.self_supervised

        if not args.test_only:
            self.criterion = get_criterion(args) if criterion is None else criterion
            self.optimizer = get_optimizer(args, model) if optimizer is None else optimizer
            self.scheduler = get_scheduler(args, self.optimizer) if scheduler is None else scheduler
            self.checkpointer = Checkpointer(args, model, self.optimizer) if checkpointer is None else checkpointer
        self.softmax = torch.nn.Softmax(dim=1)
        self.best_threshold = best_threshold

        # if 'mobilenet_v3' in self.args.model:
        if self.args.apply_swa:
            ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: 0.0001 * averaged_model_parameter + 0.9999 * model_parameter
            self.ema_model = torch.optim.swa_utils.AveragedModel(self.model, avg_fn=ema_avg)

    #===========================================================================
    def train(self):
        self.model.train()
        mode = 'train'
        self.checkpointer.writer.count_up('current_epoch')

        # Initialize result dictionaries
        losses = []
        envs_preds, envs_labels = OrderedDict(), OrderedDict()
        for env_name in self.dataloaders[mode].keys():
            envs_preds[env_name], envs_labels[env_name] = list(), list()

        total_steps = min(list(map(len, list(self.dataloaders[mode].values()))))
        for step, envs_batch in enumerate(zip(*self.dataloaders[mode].values())):
            envs_losses = list()
            rect_losses = list()
            lips_losses = list()
            cont_losses = list()
            for env_name, (images, labels) in zip(self.dataloaders[mode].keys(), envs_batch):

                if self.pretrain:
                    images_contrast = torch.cat([images[0], images[1]], 0).to(self.device)

                    _, logits_contrast = self.model(images_contrast)

                    contrast_matrix, label_contrast = contrastive_loss(logits_contrast, self.args)

                    criterion = torch.nn.CrossEntropyLoss()
                    loss = criterion(contrast_matrix, label_contrast)

                    envs_losses.append(loss)

                    if step % 500 == 0 :
                        print('\n')
                        print(torch.argmax(contrast_matrix, -1))
                    # preds = self.softmax(logits)


                else:
                    if self.self_supervised:
                        if self.args.pretrain_already:
                            images = images[0]
                        else :
                            images = torch.cat([images[0], images[1]], 0)
                            labels = torch.cat([labels]*2, 0)

                    images = images.to(self.device)
                    labels = labels.squeeze(dim=1).to(self.device)

                    # Calculate reconstruction loss
                    if self.args.oudefend:
                        oudefend_output = get_oudefend_output(self.args, self.model, images)
                        reconstruction_loss = torch.nn.functional.mse_loss(oudefend_output.view(-1), images.view(-1))
                        rect_losses.append(reconstruction_loss)

                    if self.self_supervised and not self.args.pretrain_already :
                        logits, represents = self.model(images)
                        # print(represents.shape)
                        # print(logits[0].shape, logits[1].shape)
                        contrast_matrix, label_contrast = contrastive_loss(represents, self.args)
                        cont_loss = torch.nn.CrossEntropyLoss()(contrast_matrix, label_contrast)
                        cont_losses.append(cont_loss)
                    elif self.self_supervised and self.args.pretrain_already:
                        logits, _ = self.model(images)
                    else:
                        logits = self.model(images)

                    # Calculate ERM loss
                    loss = self.criterion(logits, labels)
                    envs_losses.append(loss)

                    # Calculate Lipschitz penalty
                    if self.args.lipschitz:
                        images.requires_grad = True
                        repeated_images = images.repeat(2, 1, 1, 1, 1)
                        if self.self_supervised:
                            repeated_output = torch.stack([self.model(repeated_images[0])[0].sum(axis=0),
                                                           self.model(repeated_images[1])[0].sum(axis=0)])
                        else:
                            repeated_output = torch.stack([self.model(repeated_images[0]).sum(axis=0),
                                                           self.model(repeated_images[1]).sum(axis=0)])
                        grads = torch.autograd.grad(repeated_output, repeated_images, grad_outputs=torch.eye(2).cuda(), create_graph=True)[0]
                        Lipschitz_penalty = self.args.psi_lips * grads.abs().pow(2).mean()
                        lips_losses.append(Lipschitz_penalty)

                    # Accumulate predictions and labels
                    preds = self.softmax(logits)
                    envs_preds[env_name].extend(preds[:,-1].cpu().detach().numpy())
                    envs_labels[env_name].extend(labels.cpu().detach().numpy())

            # Total loss
            total_loss = 0.
            # Empirical Risk Minimization (ERM)
            erm_loss = torch.stack(envs_losses).mean()
            total_loss += erm_loss

            if not self.pretrain:
                # Reconstruction
                if self.args.oudefend:
                    rect_loss = torch.stack(rect_losses).mean()
                    total_loss += self.args.lambda_rect * rect_loss
                # Lipschitz penalty
                if self.args.lipschitz:
                    lips_penalty = torch.stack(lips_losses).mean()
                    total_loss += self.args.beta_lips * lips_penalty

                if self.self_supervised:
                    if self.args.pretrain_already :
                        cont_loss = 0
                    else :
                        cont_loss = torch.stack(cont_losses).mean()
                    total_loss += self.args.lambda_cont * cont_loss

            losses.append(total_loss.item())

            # Update parameters
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # if 'mobilenet_v3' in self.args.model:
            if self.args.apply_swa:
                self.ema_model.update_parameters(self.model)

            if self.scheduler is not None:
                self.scheduler.step()

            self.checkpointer.writer.count_up('current_iter')

            if self.pretrain:
                self.checkpointer.writer.write_result('pretrain_iteration_loss', total_loss.item(), self.checkpointer.writer.current_iter)
            else:
                self.checkpointer.writer.write_result('train_iteration_loss', total_loss.item(), self.checkpointer.writer.current_iter)
                self.checkpointer.writer.write_result('erm_loss', erm_loss.item(), self.checkpointer.writer.current_iter)
                # self.checkpointer.writer.write_result('rect_loss', rect_loss.item(), self.checkpointer.writer.current_iter)
                # self.checkpointer.writer.write_result('cont_loss', cont_loss.item(), self.checkpointer.writer.current_iter)

            # Print the current process
            if self.checkpointer.writer.current_iter % self.args.print_freq == 0:
                avg_loss = sum(losses)/len(losses)
                print('\r(Train) Step: [{:05d}/{:05d}] Iter: {:5d} --> cur.loss {:.3E} / avg.loss: {:.3E}'.format(step+1, total_steps,
                                                                                                                    self.checkpointer.writer.current_iter,
                                                                                                                    total_loss.item(), avg_loss), end='')
            if not self.pretrain:
                # Validate model
                if self.args.num_gpus > 0 and (self.args.local_rank % self.args.num_gpus == 0):
                    if self.checkpointer.writer.current_iter % self.args.val_freq == 0:
                        self.validate()
                        self.model.train()

            if self.self_supervised and self.pretrain:
                if self.checkpointer.writer.current_iter % self.args.val_freq == 0 :
                    self.checkpointer.writer.write_result('pretrain_loss', avg_loss, self.checkpointer.writer.current_epoch)
                    os.makedirs(self.args.CHECKPOINT_DIR, exist_ok=True)
                    torch.save(self.model.state_dict(), self.args.PRETRAIN_CHECKPOINT_PATH)
                    print(' '.join(['[Notice] Pretrain checkpoint have been saved at', self.args.PRETRAIN_CHECKPOINT_PATH]))

            # Stop training if max_iters reached
            if self.args.max_iters is not None and self.checkpointer.writer.current_iter >= self.args.max_iters:
                print('\n[Notice] The maximun iteration has been reached..')
                self.checkpointer.stop_training = True
                break

        # if self.scheduler is not None:
        #     self.scheduler.step()

        avg_loss = sum(losses)/len(losses)
        if self.self_supervised and self.pretrain:
            self.checkpointer.writer.write_result('pretrain_loss', avg_loss, self.checkpointer.writer.current_epoch)
            os.makedirs(self.args.CHECKPOINT_DIR, exist_ok=True)
            torch.save(self.model.state_dict(), self.args.PRETRAIN_CHECKPOINT_PATH)
            print(' '.join(['[Notice] Pretrain checkpoint have been saved at', self.args.PRETRAIN_CHECKPOINT_PATH]))
        else:
            self.checkpointer.writer.write_result('train_loss', avg_loss, self.checkpointer.writer.current_epoch)
            self.save_results(envs_preds, envs_labels, mode='train', avg_loss=avg_loss)

    #===========================================================================
    def validate(self):
        self.model.eval()
        mode = 'val'
        self.checkpointer.writer.write_result('val_iter', self.checkpointer.writer.current_iter)

        # Initialize result dictionaries
        losses = []
        envs_preds, envs_labels = OrderedDict(), OrderedDict()
        for env_name in self.dataloaders[mode].keys():
            envs_preds[env_name], envs_labels[env_name] = list(), list()

        total_steps = min(list(map(len, list(self.dataloaders[mode].values()))))
        with torch.no_grad():
            for env_name, dataloader in zip(self.dataloaders[mode].keys(), self.dataloaders[mode].values()):
                total_steps = len(dataloader)
                envs_losses = list()
                for step, (images, labels) in enumerate(dataloader):
                    # images = images[0].to(self.device) if self.self_supervised \
                    #                                    else images.to(self.device)
                    images = images.to(self.device)
                    labels = labels.squeeze(dim=1).to(self.device)

                    logits = self.model(images)[0] if self.self_supervised \
                                                   else self.model(images)

                    # Calculate ERM loss
                    loss = self.criterion(logits, labels)
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
        self.save_results(envs_preds, envs_labels, mode='val', avg_loss=avg_loss)



    #===========================================================================
    def test(self):
        if self.args.ensemble:
            for m in self.model:
                m.eval()
        else:
            self.model.eval()
        mode = 'test'

        envs_preds, envs_labels = OrderedDict(), OrderedDict()
        for env_name in self.dataloaders[mode].keys():
            envs_preds[env_name], envs_labels[env_name] = list(), list()

        total_steps = min(list(map(len, list(self.dataloaders[mode].values()))))
        with torch.no_grad():
            for env_name, dataloader in zip(self.dataloaders[mode].keys(), self.dataloaders[mode].values()):
                total_steps = len(dataloader)
                for step, (images, labels) in enumerate(dataloader):
                    images = images.to(self.device)
                    labels = labels.squeeze(dim=1)

                    if self.args.ensemble:
                        logits_list = []
                        for m in self.model:
                            output = m(images)
                            logits = output[0] if type(output) is tuple else output
                            logits_list.append(logits)
                        logits = torch.stack(logits_list, dim=0).mean(dim=0)
                    else:
                        output = self.model(images)
                        logits = output[0] if type(output) is tuple else output

                    # Accumulate predictions and labels
                    preds = self.softmax(logits)

                    envs_preds[env_name].extend(preds[:,-1].cpu().detach().numpy())
                    envs_labels[env_name].extend(labels.numpy())
                    print('\r(Test) Step: [{:05d}/{:05d}] {:<20}'.format(step+1, total_steps, env_name), end='')

        self.save_results(envs_preds, envs_labels, mode='test')

    #===========================================================================
    def add_phase(self, tensor_image):
        pil_image = ToPILImage()(tensor_image.cpu())
        gray_image = pil_image.convert('L')
        image_spectrum = np.fft.fft2(np.array(gray_image))
        phase = np.angle(image_spectrum)
        image_phase = np.fft.ifft2(phase).real
        image_phase = torch.from_numpy(image_phase).float().unsqueeze(dim=0)
        rgbp_image = torch.cat([tensor_image, image_phase.to(self.args.device)], dim=0)
        return rgbp_image
    #===========================================================================
    def attack_test(self, base_model):

        def fgsm(images, labels, eps):
            if self.args.zero_phase:
                phase = torch.zeros_like(images[:, 3:, :, :])

            if self.args.phase_concat:
                images = images[:, 0:3, :, :]
            torch.set_grad_enabled(True)
            images.requires_grad_()
            logits = base_model(images)
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            # error = criterion(logits.squeeze(), labels)
            error = criterion(logits, labels)
            error.backward()
            # perturbed_img = torch.clamp(images + eps*images.grad.data.sign(), 0, 1).detach()
            perturbed_img = (images + eps*images.grad.data.sign()).detach()
            images.grad.zero_()
            torch.set_grad_enabled(False)


            if self.args.phase_concat:
                if self.args.zero_phase:
                    perturbed_img = torch.cat((perturbed_img, phase), dim=1)
                else:
                    p_imgs = []
                    for p_img_idx in range(len(perturbed_img)):
                        p_imgs.append(self.add_phase(perturbed_img[p_img_idx]))
                    perturbed_img = torch.stack(p_imgs, dim=0)

            return perturbed_img

        if self.args.ensemble:
            for m in self.model:
                m.eval()
        else:
            self.model.eval()
        mode = 'test'

        envs_preds, envs_labels = OrderedDict(), OrderedDict()
        for env_name in self.dataloaders[mode].keys():
            envs_preds[env_name], envs_labels[env_name] = list(), list()

        total_steps = min(list(map(len, list(self.dataloaders[mode].values()))))

        eps_list = [0., 0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
        for eps in eps_list:
            print('------------------------')
            print('ours performace eps:', eps)
            print('------------------------')
            with torch.no_grad():
                for env_name, dataloader in zip(self.dataloaders[mode].keys(), self.dataloaders[mode].values()):
                    total_steps = len(dataloader)
                    for step, (images, labels) in enumerate(dataloader):
                        images = images.to(self.device)
                        labels = labels.squeeze(dim=1).to(self.device)
                        perturbed_img = fgsm(images, labels, eps)

                        if self.args.ensemble:
                            logits_list = []
                            for m in self.model:
                                output = m(perturbed_img)
                                logits = output[0] if type(output) is tuple else output
                                logits_list.append(logits)
                            logits = torch.stack(logits_list, dim=0).mean(dim=0)
                        else:
                            output = self.model(perturbed_img)
                            logits = output[0] if type(output) is tuple else output

                        # Accumulate predictions and labels
                        preds = self.softmax(logits)
                        envs_preds[env_name].extend(preds[:,-1].cpu().detach().numpy())
                        envs_labels[env_name].extend(labels.cpu().numpy())
                        print('\r(Test) Step: [{:05d}/{:05d}] {:<20}'.format(step+1, total_steps, env_name), end='')

            self.save_results(envs_preds, envs_labels, mode='test')


            print('-----------------------')
            print('base_model performace eps:', eps)
            print('-----------------------')
            with torch.no_grad():
                for env_name, dataloader in zip(self.dataloaders[mode].keys(), self.dataloaders[mode].values()):
                    total_steps = len(dataloader)
                    for step, (images, labels) in enumerate(dataloader):
                        images = images.to(self.device)
                        labels = labels.squeeze(dim=1).to(self.device)
                        perturbed_img = fgsm(images, labels, eps)

                        output = base_model(perturbed_img[:, :3, :,:])
                        logits = output[0] if type(output) is tuple else output

                        # Accumulate predictions and labels
                        preds = self.softmax(logits)
                        envs_preds[env_name].extend(preds[:,-1].cpu().detach().numpy())
                        envs_labels[env_name].extend(labels.cpu().numpy())
                        print('\r(Test) Step: [{:05d}/{:05d}] {:<20}'.format(step+1, total_steps, env_name), end='')
            self.save_results(envs_preds, envs_labels, mode='test')

    #===========================================================================
    def save_results(self, envs_preds, envs_labels, mode, avg_loss=None):
        results = OrderedDict()
        total_preds = list()
        total_labels = list()
        total_unseen_preds = list()
        total_unseen_labels = list()

        train_domains = self.args.domain_list

        # Split seen/unseen dataset
        for env_name in envs_labels.keys():
            e_preds = torch.Tensor(envs_preds[env_name])
            e_labels = torch.Tensor(envs_labels[env_name])

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

            if mode == 'val':
                # get the best threshold
                fpr, tpr, thresholds = roc_curve(np.array(t_labels), np.array(t_preds))
                ix = np.argmax(tpr - fpr)
                self.best_threshold = thresholds[ix]
                self.checkpointer.writer.best_threshold = self.best_threshold

            total_avg_acc = mean_accuracy(t_preds, t_labels, self.best_threshold)
            total_real_acc = mean_accuracy(t_preds[t_labels==0], t_labels[t_labels==0], self.best_threshold)
            total_fake_acc = mean_accuracy(t_preds[t_labels==1], t_labels[t_labels==1], self.best_threshold)
            total_roc_auc = roc_auc_score(np.array(t_labels), np.array(t_preds))

            if not self.args.test_only and self.checkpointer is not None:
                self.checkpointer.writer.write_result('{}_acc'.format(mode), total_avg_acc, self.checkpointer.writer.current_iter, tb_only=(mode=='test'))
                self.checkpointer.writer.write_result('{}_roc_auc'.format(mode), total_roc_auc, self.checkpointer.writer.current_iter, tb_only=(mode=='test'))
                self.checkpointer.writer.write_result('{}_iteration_acc'.format(mode), total_avg_acc, self.checkpointer.writer.current_iter, tb_only=True)
                self.checkpointer.writer.write_result('{}_iteration_roc_auc'.format(mode), total_roc_auc, self.checkpointer.writer.current_iter, tb_only=True)
            results['Total (seen)'] = {'avg_acc': total_avg_acc, 'real_acc': total_real_acc, 'fake_acc': total_fake_acc,
                                'num_real': len(t_labels[t_labels==0]), 'num_fake': len(t_labels[t_labels==1]), 'roc_auc': total_roc_auc}
        else:
            if not self.args.test_only and self.checkpointer is not None:
                self.checkpointer.writer.write_result('{}_acc'.format(mode), 0, self.checkpointer.writer.current_iter, tb_only=(mode=='test'))
                self.checkpointer.writer.write_result('{}_roc_auc'.format(mode), 0, self.checkpointer.writer.current_iter, tb_only=(mode=='test'))

        # Calculate Acc. and ROC AUC score for all (unseen) data
        if mode!='train':
            if len(total_unseen_preds)>0:
                t_preds = torch.Tensor(total_unseen_preds)
                t_labels = torch.Tensor(total_unseen_labels)
                total_avg_acc = mean_accuracy(t_preds, t_labels, self.best_threshold)
                total_real_acc = mean_accuracy(t_preds[t_labels==0], t_labels[t_labels==0], self.best_threshold)
                total_fake_acc = mean_accuracy(t_preds[t_labels==1], t_labels[t_labels==1], self.best_threshold)
                total_roc_auc = roc_auc_score(np.array(t_labels), np.array(t_preds))
                if not self.args.test_only and self.checkpointer is not None:
                    if mode=='val':
                        self.checkpointer.writer.write_result('val_acc_us', total_avg_acc, self.checkpointer.writer.current_iter)
                        self.checkpointer.writer.write_result('val_roc_auc_us', total_roc_auc, self.checkpointer.writer.current_iter)
                    self.checkpointer.writer.write_result('{}_iteration_acc_us'.format(mode), total_avg_acc, self.checkpointer.writer.current_iter, tb_only=True)
                    self.checkpointer.writer.write_result('{}_iteration_roc_auc_us'.format(mode), total_roc_auc, self.checkpointer.writer.current_iter, tb_only=True)
                results['Total (unseen)'] = {'avg_acc': total_avg_acc, 'real_acc': total_real_acc, 'fake_acc': total_fake_acc,
                                    'num_real': len(t_labels[t_labels==0]), 'num_fake': len(t_labels[t_labels==1]), 'roc_auc': total_roc_auc}
            else:
                if mode=='val':
                    self.checkpointer.writer.write_result('val_acc_us', 0, self.checkpointer.writer.current_iter)
                    self.checkpointer.writer.write_result('val_roc_auc_us', 0, self.checkpointer.writer.current_iter)

        # Calcualte each environment's socres
        for env_name in envs_labels.keys():
            e_preds = torch.Tensor(envs_preds[env_name])
            e_labels = torch.Tensor(envs_labels[env_name])
            avg_acc = mean_accuracy(e_preds, e_labels, self.best_threshold)
            real_acc = mean_accuracy(e_preds[e_labels==0], e_labels[e_labels==0], self.best_threshold)
            fake_acc = mean_accuracy(e_preds[e_labels==1], e_labels[e_labels==1], self.best_threshold)
            roc_auc = roc_auc_score(np.array(envs_labels[env_name]), np.array(envs_preds[env_name]))

            if not self.args.test_only and self.checkpointer is not None:
                self.checkpointer.writer.write_result('{}_iteration_acc_{}'.format(mode, env_name), avg_acc, self.checkpointer.writer.current_iter, tb_only=True)

            results[env_name] = {'avg_acc': avg_acc, 'real_acc': real_acc, 'fake_acc': fake_acc, 'roc_auc': roc_auc,
                                 'num_real': len(e_labels[e_labels==0]), 'num_fake': len(e_labels[e_labels==1])}

        # Print results
        if mode != 'test' and self.args.dataset is None:
            print('-----> {} avg.loss: {:.3E} / acc.: {:6.2f} (R {:6.2f} / F {:6.2f}) / roc auc score: {:.4f}'.format(mode, avg_loss, total_avg_acc*100, total_real_acc*100, total_fake_acc*100, total_roc_auc))
        if mode != 'train':
            print('-----> environment results (threshold: {:.3f})'.format(self.best_threshold))
            for env_name, result in results.items():
                if 'Total' not in env_name:
                    if env_name in train_domains:
                        print('{:<15}{:<10}acc.: {:6.2f} (R {:6.2f} / F {:6.2f}){:2}[{:4d}/{:4d}] / roc auc score: {:.4f}'.format(env_name, '(seen)', result['avg_acc']*100,
                                result['real_acc']*100, result['fake_acc']*100, '', result['num_real'], result['num_fake'], result['roc_auc']))
                    else:
                        print('{:<15}{:<10}acc.: {:6.2f} (R {:6.2f} / F {:6.2f}){:2}[{:4d}/{:4d}] / roc auc score: {:.4f}'.format(env_name, '(unseen)', result['avg_acc']*100,
                                result['real_acc']*100, result['fake_acc']*100, '', result['num_real'], result['num_fake'], result['roc_auc']))
                else:
                    if '(seen)' in env_name:
                        print('{:<15}{:<10}acc.: {:6.2f} (R {:6.2f} / F {:6.2f}){:2}[{:4d}/{:4d}] / roc auc score: {:.4f}'.format(env_name, '', result['avg_acc']*100,
                            result['real_acc']*100, result['fake_acc']*100, '', result['num_real'], result['num_fake'], result['roc_auc']))
                    if 'unseen' in env_name:
                        print('{:<15}{:<10}acc.: {:6.2f} (R {:6.2f} / F {:6.2f}){:2}[{:4d}/{:4d}] / roc auc score: {:.4f}'.format(env_name, '', result['avg_acc']*100,
                            result['real_acc']*100, result['fake_acc']*100, '', result['num_real'], result['num_fake'], result['roc_auc']))

        if mode == 'test':
            for env_name, result in results.items():
                if env_name == 'Total (seen)':
                    print('{:.4f}'.format(result['roc_auc']), end='\t')
                    break
            for env_name, result in results.items():
                if 'Total' not in env_name:
                    print('{:.4f}'.format(result['roc_auc']), end='\t')
            for env_name, result in results.items():
                if env_name == 'Total (unseen)':
                    print('{:.4f}'.format(result['roc_auc']), end='\t')
                    break

        if mode=='val':
            score = results['Total (seen)']['roc_auc']
            self.checkpointer.save_checkpoint(score=score, verbose=self.args.verbose)
