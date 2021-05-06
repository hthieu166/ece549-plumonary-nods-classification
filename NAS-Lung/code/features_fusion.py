#Adding Gradient-boosting-classifier
from sklearn.ensemble import GradientBoostingClassifier as gbt
import numpy as np
import ipdb
class FeaturesFusion():
    def __init__(self, is_enababled = True):
        self.is_enababled        = is_enababled
        self.gbt_depth = 1
        self.best_acc  = 0
        self.train_deep_feats    = []
        self.train_low_lvl_feats = []
        self.train_targets       = []
    
    def reset(self):
        if self.is_enababled == False:
            return
        self.train_deep_feats    = []
        self.train_low_lvl_feats = []
        self.train_targets       = []
        self.test_deep_feats    = []
        self.test_low_lvl_feats = []
        self.test_targets       = []

    def add_train_features(self, outputs, feat, targets):
        if self.is_enababled == False:
            return
        self.train_deep_feats.append(torch.stack(outputs[2]).cpu().detach().numpy())
        self.train_targets.append(targets.cpu().detach().numpy())
        self.train_low_lvl_feats.append(feat)
    
    def add_test_features(self, outputs, feat, targets):
        if self.is_enababled == False:
            return
        self.test_deep_feats.append(torch.stack(outputs[2]).cpu().detach().numpy())
        self.test_targets.append(targets.cpu().detach().numpy())
        self.test_low_lvl_feats.append(feat)

    def fusion_feats(self, deep_feat, low_lvl_feat):
        if self.is_enababled == False:
            return None
        return np.concatenate([deep_feat[0,...], low_lvl_feat], axis=1)

    def fit(self):
        if self.is_enababled == False:
            return None
        self.train_deep_feats   = np.hstack(self.train_deep_feats)
        self.train_low_lvl_feats= np.concatenate(self.train_low_lvl_feats, axis=0) 
        self.train_targets      = np.hstack(self.train_targets)

        self.m = gbt(max_depth= self.gbt_depth, random_state=0)
        train_feat = self.fusion_feats(deep_feat, low_lvl_feat)
        print("retraining")
        self.m.fit(train_feat, targets)
        return np.mean(self.m.predict(train_feat) == targets)

    def predict(self):
        if self.is_enababled == False:
            return None
        self.test_deep_feats   = np.hstack(test_deep_feats)
        self.test_low_lvl_feats= np.concatenate(test_low_lvl_feats, axis=0) 
        self.test_targets      = np.hstack(test_targets)
        test_feat = self.fusion_feats(self.test_deep_feats, self.test_low_lvl_feats)
        return self.m.predict(test_feat)