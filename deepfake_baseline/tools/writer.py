# Copyright © 2020 Jungwon Choi <jungwon.choi@kaist.ac.kr>
import os
import torch
from torch.utils.tensorboard import SummaryWriter


#===============================================================================
class ResultWriter():
    def __init__(self, args, score_higher_better=True):
        if not args.norecord:
            self.tensorboard_writer     = SummaryWriter(args.logs_path)
        self.norecord                   = args.norecord
        self.current_epoch              = 0
        self.current_iter               = 0
        self.current_lr                 = None
        self.train_iteration_loss       = []
        self.train_loss                 = []
        self.train_acc                  = []
        self.train_roc_auc              = []
        self.val_iter                   = []
        self.val_loss                   = []
        self.val_acc                    = []
        self.val_roc_auc                = []
        self.high_score                 = score_higher_better
        if self.high_score:
            self.best_score             = 0.
            self.best_score_us          = 0.
        else:
            self.best_score             = float('inf')
            self.best_score_us          = float('inf')

    def write_result(self, name, value, idx=None, tb_only=False):
        if self.norecord: return
        if idx is not None:
            self.tensorboard_writer.add_scalar(name, value, idx)
            if tb_only: return
        if eval("type(self.{}) is list".format(name)):
            exec("self.{}.append(value)".format(name))
        else:
            exec("self.{}={}".format(name, value))

    def write_image(self, name, img, idx):
        if self.norecord: return
        self.tensorboard_writer.add_image(name, img, idx)

    def count_up(self, name):
        exec("self.{}+=1".format(name))

    def set_variable(self, name, value):
        setattr(self, name, value)

#===============================================================================
class Checkpointer():
    def __init__(self, args, model, optimizer=None, scheduler=None):
        self.writer = ResultWriter(args, score_higher_better=True)
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.previous_score = None
        self.stop_tolerance = 1e-3
        self.stop_patience = 0
        self.stop_training = False

        if args.resume:
            assert os.path.exists(args.RESUME_CHECKPOINT_PATH), 'No such checkpoint file!'
            checkpoint_dict = torch.load(args.RESUME_CHECKPOINT_PATH, map_location=args.device)
            self.model.load_state_dict(checkpoint_dict['model_state_dict'])
            print('[Notice] Load resume model: {} epochs ({} iters)'.format(checkpoint_dict['current_epoch'],
                                                                            checkpoint_dict['current_iter']))
            if (self.optimizer is not None) and ('optimizer_state_dict' in checkpoint_dict):
                self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            if (self.scheduler is not None) and ('scheduler_state_dict' in checkpoint_dict):
                self.scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
            if (self.writer is not None) and (checkpoint_dict['current_iter'] > 0):
                self.writer.current_epoch           = checkpoint_dict['current_epoch']
                self.writer.current_iter            = checkpoint_dict['current_iter']
                self.writer.current_lr              = checkpoint_dict['current_lr']
                self.writer.train_iteration_loss    = checkpoint_dict['train_iteration_loss']
                self.writer.train_loss              = checkpoint_dict['train_loss']
                self.writer.train_acc               = checkpoint_dict['train_acc']
                self.writer.train_roc_auc           = checkpoint_dict['train_roc_auc']
                self.writer.val_iter                = checkpoint_dict['val_iter']
                self.writer.val_loss                = checkpoint_dict['val_loss']
                self.writer.val_acc                 = checkpoint_dict['val_acc']
                self.writer.val_roc_auc             = checkpoint_dict['val_roc_auc']
                self.writer.best_score              = checkpoint_dict['best_score']


    def save_checkpoint(self, score=None, save_best=True, without_val=False, verbose=False):
        checkpoint_dict = {}
        checkpoint_dict['model_state_dict']          = self.model.state_dict() # if not self.args.multi_gpu else self.model.module.state_dict()
        if self.optimizer is not None:
            checkpoint_dict['optimizer_state_dict']  = self.optimizer.state_dict()
        if self.scheduler is not None:
            checkpoint_dict['scheduler_state_dict']  = self.scheduler.state_dict()
        if self.writer is not None:
            checkpoint_dict['current_epoch']         = self.writer.current_epoch
            checkpoint_dict['current_iter']          = self.writer.current_iter
            checkpoint_dict['current_lr']            = self.writer.current_lr
            checkpoint_dict['train_iteration_loss']  = self.writer.train_iteration_loss
            checkpoint_dict['train_loss']            = self.writer.train_loss
            checkpoint_dict['train_acc']             = self.writer.train_acc
            checkpoint_dict['train_roc_auc']         = self.writer.train_roc_auc
            checkpoint_dict['val_iter']              = self.writer.val_iter
            checkpoint_dict['val_loss']              = self.writer.val_loss
            checkpoint_dict['val_acc']               = self.writer.val_acc
            checkpoint_dict['val_roc_auc']           = self.writer.val_roc_auc
            checkpoint_dict['best_score']            = self.writer.best_score

        os.makedirs(self.args.CHECKPOINT_DIR, exist_ok=True)

        if not os.path.isfile(self.args.ARGS_INFO_PATH):
            with open(self.args.ARGS_INFO_PATH, 'a') as f:
                f.write("==========       CONFIG      =============\n")
                for arg, content in self.args.__dict__.items():
                    f.write("{:35s}\t: {:}\n".format(arg, content))
                f.write("==========     CONFIG END    =============\n")

        score = self.writer.val_roc_auc[-1] if score is None else score
        if self.args.earlystop:
            if self.previous_score is None:
                self.previous_score = score
            else:
                if abs(score-self.previous_score)/score*100 < self.stop_tolerance:
                    self.stop_patience += 1
                else:
                    self.stop_patience = 0
                self.previous_score = score
            if self.stop_patience >= 10:
                print('\n[Notice] The validation score does not change for 10 consecutive checks. (Early Stop)')
                self.stop_training = True
        try:
            torch.save(checkpoint_dict, self.args.LAST_CHECKPOINT_PATH)
            if self.args.verbose: print(' '.join(['[Notice] Checkpoint have been saved at', self.args.LAST_CHECKPOINT_PATH]))

            if not without_val:
                # Save best checkpoint
                if save_best:
                    if score > self.writer.best_score:
                        self.writer.best_score = score
                        torch.save(checkpoint_dict, self.args.BEST_CHECKPOINT_PATH)
                        print(' '.join(['[Notice] Best score checkpoint have been saved at', self.args.BEST_CHECKPOINT_PATH]))

            if (self.args.max_iters is not None and self.writer.current_iter >= self.args.max_iters) or self.stop_training:
                torch.save(checkpoint_dict, self.args.FINAL_CHECKPOINT_PATH)
                print(' '.join(['[Notice] Final checkpoint have been saved at', self.args.FINAL_CHECKPOINT_PATH]))

        except KeyboardInterrupt:
            # KeyboardInterrupt during checkpoint saving
            torch.save(checkpoint_dict, self.args.LAST_CHECKPOINT_PATH)
            print(' '.join(['[Interrupt] Checkpoint have been saved at', self.args.LAST_CHECKPOINT_PATH]))
            exit()
