import yaml
import os.path as osp

class Config():
    def __init__(self, cfg_file):
        assert osp.exists(cfg_file), cfg_file + " does not exist!"
        self._config = Config.__load_yml(cfg_file)
        _mod_config = self._config["model"]
        self.model_name   = _mod_config["name"]
        self.model_config = _mod_config["params"]
        self.dataset_params = self._config["dataset"]
        self.train_params   = self._config["train_params"] 
        self.experiment_id= osp.basename(cfg_file).replace(".yml", "") 
        print(self.model_config)
    
    @staticmethod
    def __load_yml(file_path):
        with open (file_path) as fi:
            return yaml.load(fi, Loader=yaml.FullLoader)

if __name__ == "__main__":
    Config("configs/sample_config.yml")
    
