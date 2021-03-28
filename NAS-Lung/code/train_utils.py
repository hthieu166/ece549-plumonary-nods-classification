import logging
from torch.utils.tensorboard import SummaryWriter
import torch
import os.path as osp
import os

class TrainUtils():
    def __init__(self, log_dir, model_selection_criteria = lambda x,y: x > y, ckpt_every = 10):
        self.log_dir    = log_dir
        self.ckpt_every = ckpt_every
        os.makedirs(log_dir, exist_ok = True)
        self.tensorboard_dir = osp.join(log_dir, "tensorboard")
        os.makedirs(self.tensorboard_dir, exist_ok = True)
        # Set up log writer
        self.writer  = SummaryWriter(self.tensorboard_dir)
        logging.basicConfig(filename=osp.join(self.log_dir, 'log.txt'), level=logging.INFO)
        # Model selection criteria
        self.model_selection_criteria = model_selection_criteria
        self.best_acc = -1

    def log(self, message):
        print(message)
        logging.info(message)

    def resume_from_ckpt(self, n_epoch):
        model_path = osp.join(self.log_dir, self.__get_best_checkpoint_name(n_epoch))
        assert osp.exists(model_path), "Error: no checkpoint found!"
        model = torch.load(model_path)
        self.best_acc = model["eval"]["best_acc"]
        return model
    

    def __get_checkpoint_name(self, n_epoch):
        return "{:04}.model".format(n_epoch)
    
    def __get_best_checkpoint_name(self):
        return "best.model"

    def save_checkpoint_every(self, obj, cur_epoch):
        if cur_epoch % self.ckpt_every == 0:
            self._save_checkpoint(obj, self.__get_checkpoint_name(cur_epoch))

    def _save_checkpoint(self, obj, checkpoint_name):
        torch.save(obj, osp.join(self.log_dir, checkpoint_name))
    
    def __eval_to_str(self, n_epoch, eval_dict, mode):
        str = "[Epoch {:05d} {}]".format(n_epoch, mode)
        for k in eval_dict.keys():
            str += "\t{} = {:.04f}".format(k, eval_dict[k])
        return str

    def add_train_info(self, n_epoch, info_dict):
        self.log_tensorboard("acc", info_dict["acc"], n_epoch, mode = "train")
        self.log_tensorboard("loss", info_dict["loss"], n_epoch, mode = "train")
        self.log(self.__eval_to_str(n_epoch, info_dict, mode = "train"))

    def add_test_ckpt(self, n_epoch, state):
        info_dict = state["eval"]
        is_better = self.model_selection_criteria(info_dict["acc"], self.best_acc)
        if (is_better):
            self.best_acc = info_dict["acc"]
        #Append best_acc, epoch to eval dict
        info_dict["best_acc"] = self.best_acc
        self.log_tensorboard("acc", info_dict["acc"], n_epoch, mode = "test")
        self.log_tensorboard("tpr", info_dict["tpr"], n_epoch, mode = "test")
        self.log_tensorboard("fpr", info_dict["fpr"], n_epoch, mode = "test")
        self.log_tensorboard("best_acc",  info_dict["best_acc"], n_epoch, mode = "test")
        self.log(self.__eval_to_str(n_epoch, info_dict, mode = "test"))

        #Save checkpoint after a fixed period
        self.save_checkpoint_every(info_dict, n_epoch)
        #Save best checkpoint
        if (is_better):
            self._save_checkpoint(info_dict, self.__get_best_checkpoint_name())
        

    def log_tensorboard(self, tag, value, step, mode="train"):
        self.writer.add_scalar("{}/{}".format(tag,mode), value, step)
        
        
        
