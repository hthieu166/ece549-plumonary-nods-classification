#Adding Gradient-boosting-classifier
from sklearn.ensemble import GradientBoostingClassifier as gbt
import numpy as np
import ipdb
class FeaturesFusion():
    def __init__(self):
        self.gbt_depth = 1
        self.best_acc  = 0
    
    def fusion_feats(self, deep_feat, low_lvl_feat):
        return np.concatenate([deep_feat[0,...], low_lvl_feat], axis=1)

    def fit(self, deep_feat, low_lvl_feat, targets):
        self.m = gbt(max_depth= self.gbt_depth, random_state=0)
        train_feat = self.fusion_feats(deep_feat, low_lvl_feat)
        print("retraining")
        self.m.fit(train_feat, targets)
        return np.mean(self.m.predict(train_feat) == targets)

    def predict(self, deep_feat, low_lvl_feat):
        test_feat = self.fusion_feats(deep_feat, low_lvl_feat)
        return self.m.predict(test_feat)