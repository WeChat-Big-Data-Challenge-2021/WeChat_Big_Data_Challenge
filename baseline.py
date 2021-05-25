# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
# import tensorflow.compat.v1 as tf
# from tensorflow import feature_column as fc
from comm import ACTION_LIST, STAGE_END_DAY, FEA_COLUMN_LIST, ROOT_PATH
from evaluation import uAUC, compute_weighted_score
import sys
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold,KFold
from copy import deepcopy
model_checkpoint_dir = './data/model'

SEED = 100

class Lightgbm(object):
    def __init__(self, stage, action, k_fold):
        """
        :param linear_feature_columns: List of tensorflow feature_column
        :param dnn_feature_columns: List of tensorflow feature_column
        :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
        :param action: String. Including "read_comment"/"like"/"click_avatar"/"favorite"/"forward"/"comment"/"follow"
        """
        super(Lightgbm, self).__init__()
        self.num_epochs_dict = {"read_comment": 1, "like": 1, "click_avatar": 1, "favorite": 1, "forward": 1,
                                "comment": 1, "follow": 1}
        self.bst = None
        self.stage = stage
        self.action = action
        self.k_fold = k_fold

    def pre_data(self):
        if self.stage in ["online_train", "offline_train"]:
            # 训练集，每个action一个文件
            action = self.action
        elif self.stage in ["evaluate"]:
            # 测试集，所有action在同一个文件
            action = "all"
        elif self.stage in ["submit"]:
            action = "all"
            file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action=action,
                                                                      day=STAGE_END_DAY[self.stage])
            stage_dir = os.path.join(ROOT_PATH, self.stage, file_name)
            df = pd.read_csv(stage_dir)
            X = df
            self.x_test = X
            self.id = df[["userid", "feedid"]]
            return
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action=action,
                                                                      day=STAGE_END_DAY[self.stage])
        stage_dir = os.path.join(ROOT_PATH, self.stage, file_name)
        df = pd.read_csv(stage_dir)
        y = np.array(df[self.action])
        if self.stage in ["evaluate"]:
            X = np.array(df.drop(ACTION_LIST, axis=1))
        else:
            X = np.array(df.drop(self.action, axis=1))
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.10, random_state=SEED)
        
        self.sfolder = KFold(n_splits=self.k_fold, random_state=SEED)

    def load_model(self):
        self.model_list = []
        if self.stage in ["evaluate", "offline_train"]:
            stage = "offline_train"
        else:
            stage = "online_train"
        self.model_checkpoint_stage_dir = os.path.join(model_checkpoint_dir, stage, self.action)
        if not os.path.exists(self.model_checkpoint_stage_dir):
            # 如果模型目录不存在，则创建该目录
            os.makedirs(self.model_checkpoint_stage_dir)
        elif self.stage in ["online_train", "offline_train"]:
            # 训练时如果模型目录已存在，则清空目录
            del_file(self.model_checkpoint_stage_dir)
        else:
            for i in range(self.k_fold):
                self.model_list.append(lgb.Booster(model_file=os.path.join(self.model_checkpoint_stage_dir, 'model_' + str(i) + '.txt')))

    def train(self):
        """
        训练单个行为的模型
        """
        param = {'num_leaves':31, 'num_trees':10, 'objective':'binary'}
        param['metric'] = 'binary_logloss'
        num_round = 10
        for index, (train_index, val_index) in enumerate(self.sfolder.split(self.x_train, self.y_train)):
            x_train = self.x_train[train_index]
            y_train = self.y_train[train_index]
            x_val = self.x_train[val_index]
            y_val = self.y_train[val_index]
            train_data = lgb.Dataset(data=x_train,label=y_train)
            val_data = lgb.Dataset(data=x_val,label=y_val)
            bst = lgb.train(param, train_data, num_round, valid_sets=[val_data])
            print(bst)
            bst.save_model(os.path.join(self.model_checkpoint_stage_dir, 'model_' + str(index) + '.txt'), num_iteration=bst.best_iteration)
            self.model_list.append(deepcopy(bst))

    def evaluate(self):
        """
        评估单个行为的uAUC值
        """
        if self.stage in ["online_train", "offline_train"]:
            # 训练集，每个action一个文件
            action = self.action
        else:
            # 测试集，所有action在同一个文件
            action = "all"
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action=action,
                                                                      day=STAGE_END_DAY[self.stage])
        evaluate_dir = os.path.join(ROOT_PATH, self.stage, file_name)
        df = pd.read_csv(evaluate_dir)

        y_pred = []
        for i in range(self.k_fold):
            y_pred.append(self.model_list[i].predict(self.x_test, num_iteration=self.model_list[i].best_iteration))
            # ypred = self.bst.predict(self.x_test, num_iteration=self.bst.best_iteration)
        y_pred = np.array(y_pred)
        y_pred = np.mean(y_pred, axis = 0)

        userid_list = df['userid'].astype(str).tolist()
        uauc = uAUC(self.y_test, y_pred, userid_list)
        return y_pred, uauc
    
    def predict(self):
        '''
        预测单个行为的发生概率
        '''
        t = time.time()

        y_pred = []
        for i in range(self.k_fold):
            y_pred.append(self.model_list[i].predict(self.x_test, num_iteration=self.model_list[i].best_iteration))
        y_pred = np.array(y_pred)
        logits = np.mean(y_pred, axis = 0)

        # logits = self.bst.predict(self.x_test, num_iteration=self.bst.best_iteration)
        # 计算2000条样本平均预测耗时（毫秒）
        ts = (time.time()-t)*1000.0/len(self.id)*2000.0
        return self.id, logits, ts

def del_file(path):
    '''
    删除path目录下的所有内容
    '''
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            print("del: ", c_path)
            os.remove(c_path)

def main():
    t = time.time() 
    stage = sys.argv[1]
    print('Stage: %s'%stage)
    eval_dict = {}
    predict_dict = {}
    predict_time_cost = {}
    ids = None
    
    for action in ACTION_LIST:
        print("Action:", action)
        model = Lightgbm(stage, action, 5)
        model.pre_data()
        if stage in ["online_train", "offline_train"]:
            # 训练 并评估
            model.load_model()
            model.train()
            logits, action_uauc = model.evaluate()
            eval_dict[action] = action_uauc

        if stage == "evaluate":
            # 评估线下测试集结果，计算单个行为的uAUC值，并保存预测结果
            model.load_model()
            logits, action_uauc = model.evaluate()
            eval_dict[action] = action_uauc
            predict_dict[action] = logits

        if stage == "submit":
            model.load_model()
            # 预测线上测试集结果，保存预测结果
            ids, logits, ts = model.predict()
            predict_time_cost[action] = ts
            predict_dict[action] = logits
    if stage in ["evaluate", "offline_train", "online_train"]:
        # 计算所有行为的加权uAUC
        print(eval_dict)
        weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                       "comment": 1, "follow": 1}
        weight_auc = compute_weighted_score(eval_dict, weight_dict)
        print("Weighted uAUC: ", weight_auc)


    if stage in ["evaluate", "submit"]:
        # 保存所有行为的预测结果，生成submit文件
        actions = pd.DataFrame.from_dict(predict_dict)
        print("Actions:", actions)
        ids[["userid", "feedid"]] = ids[["userid", "feedid"]].astype(int)
        res = pd.concat([ids, actions], sort=False, axis=1)
        # 写文件
        file_name = "submit_" + str(int(time.time())) + ".csv"
        submit_file = os.path.join(ROOT_PATH, stage, file_name)
        print('Save to: %s'%submit_file)
        res.to_csv(submit_file, index=False)

    if stage == "submit":
        print('不同目标行为2000条样本平均预测耗时（毫秒）：')
        print(predict_time_cost)
        print('单个目标行为2000条样本平均预测耗时（毫秒）：')
        print(np.mean([v for v in predict_time_cost.values()]))
    print('Time cost: %.2f s'%(time.time()-t))


if __name__ == "__main__":
    main()
    
