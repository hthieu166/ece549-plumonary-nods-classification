import yaml
import os.path as osp

class Config():
    def __init__(self, cfg_file):
        assert osp.exists(cfg_file), cfg_file + " does not exist!"
        #Read main config
        self._config = Config.__load_yml(cfg_file)
        self.experiment_id= osp.basename(cfg_file).replace(".yml", "") 
        #Model
        _mod_config = self._config["model"]
        self.model_name   = _mod_config["name"]
        if "params" in _mod_config:
            self.model_config = _mod_config["params"]
        self.dataset_params = self._config["dataset"]
        #Train params
        self.train_params   = self._config["train_params"] 
        #Loss
        if type(self._config["loss"]) == str:
            self.loss_name = self._config["loss"]
        else:
            self.loss_name = self._config["loss"]["name"]
        self.loss = self._config["loss"]

    @staticmethod
    def __load_yml(file_path):
        with open (file_path) as fi:
            return yaml.load(fi, Loader=yaml.FullLoader)

if __name__ == "__main__":
    Config("configs/sample_config.yml")
    
