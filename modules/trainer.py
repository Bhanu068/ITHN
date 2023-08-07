import logging
import os
from abc import abstractmethod

import torch
from numpy import inf
import math


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        self.args = args

        if not os.path.exists(f"{self.args.save_dir}/{self.args.model_name}"):
            os.makedirs(f"{self.args.save_dir}/{self.args.model_name}")

        logging.basicConfig(filename=f"{self.args.save_dir}/{self.args.model_name}/{self.args.model_name}.log",
                            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', 
                            filemode='a',
                            level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.mse_loss = torch.nn.MSELoss()
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = f"{self.args.save_dir}/{self.args.model_name}"

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.logger.info(f"{self.args}")

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        if self.args.resume == "best":
            self._test_epoch()
        else:
            not_improved_count = 0
            for epoch in range(self.start_epoch, self.epochs + 1):
                lambd = math.exp(-self.args.alpha * epoch)
                result = self._train_epoch(epoch, lambd)

                # save logged informations into log dict
                log = {'epoch': epoch}
                log.update(result)
                self._record_best(log)

                # print logged informations to the screen
                for key, value in log.items():
                    self.logger.info('\t{:15s}: {}'.format(str(key), str(value)))

                # evaluate model performance according to configured metric, save best checkpoint as model_best
                best = False
                if self.mnt_mode != 'off':
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < self.mnt_best) or \
                                (self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best)
                    except KeyError:
                        self.logger.warning(
                            "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                                self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        best = True
                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        self.logger.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                            self.early_stop))
                        break

                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch, save_best=best)

            self.args.resume = "best"
            self._test_epoch()

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] < self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] > self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

    def _print_best(self):
        self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), str(value)))

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, f"{self.args.model_name}.pth")
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, f"best_{self.args.model_name}.pth")
            torch.save(state, best_path)
            self.logger.info(f"Saving current best: best_{self.args.model_name}.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
    

class TrainerITHN(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,
                 val_dataloader, test_dataloader):
        super(TrainerITHN, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch, lambd):

        self.logger.info('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
        train_loss = 0
        self.model.train()
        for batch_idx, (image, neg_image, reports_ids, neg_reports_ids, reports_masks, neg_reports_masks) in enumerate(self.train_dataloader):

            image, neg_image, reports_ids, neg_reports_ids, reports_masks, neg_reports_masks = image.to(self.device), \
                neg_image.to(self.device), reports_ids.to(self.device), neg_reports_ids.to(self.device), \
                    reports_masks.to(self.device), neg_reports_masks.to(self.device)
            
            output, l_cp, l_cs = self.model(image, neg_image, reports_ids, reports_masks, neg_reports_ids, neg_reports_masks, lambd = lambd, mode = 'train')

            loss = self.criterion(output, reports_ids, reports_masks)
            loss = 0.7 * loss + 0.1 * l_cp + 0.2 * l_cs
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_period == 0:
                self.logger.info('[{}/{}] Step: {}/{}, Training Loss: {:.5f}.'
                                 .format(epoch, self.epochs, batch_idx, len(self.train_dataloader),
                                         train_loss / (batch_idx + 1)))

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images, _, reports_ids, _, _, _) in enumerate(self.val_dataloader):
                images, reports_ids = images.to(self.device), reports_ids.to(self.device)

                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.lr_scheduler.step()

        return log
    
    def _test_epoch(self):
        if self.args.resume == "best":
            self._resume_checkpoint(os.path.join(self.checkpoint_dir, f"best_{self.args.model_name}.pth"))
        if self.args.resume == "last":
            self._resume_checkpoint(os.path.join(self.checkpoint_dir, f"{self.args.model_name}.pth"))
        self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(self.start_epoch - 1, self.epochs))
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images, _, reports_ids, _, _, _) in enumerate(self.test_dataloader):
                images, reports_ids = images.to(self.device), reports_ids.to(self.device)
                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log = {'test_' + k: v for k, v in test_met.items()}
            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), str(value)))