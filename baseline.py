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
from sklearn.metrics import auc
model_checkpoint_dir = './data/model'
# flags = tf.app.flags
# FLAGS = flags.FLAGS

# flags.DEFINE_string('model_checkpoint_dir', './data/model', 'model dir')
# flags.DEFINE_string('root_path', './data/', 'data dir')
# flags.DEFINE_integer('batch_size', 128, 'batch_size')
# flags.DEFINE_integer('embed_dim', 10, 'embed_dim')
# flags.DEFINE_float('learning_rate', 0.1, 'learning_rate')
# flags.DEFINE_float('embed_l2', None, 'embedding l2 reg')

SEED = 2021



class WideAndDeep(object):

    def __init__(self, linear_feature_columns, dnn_feature_columns, stage, action):
        """
        :param linear_feature_columns: List of tensorflow feature_column
        :param dnn_feature_columns: List of tensorflow feature_column
        :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
        :param action: String. Including "read_comment"/"like"/"click_avatar"/"favorite"/"forward"/"comment"/"follow"
        """
        super(WideAndDeep, self).__init__()
        self.num_epochs_dict = {"read_comment": 1, "like": 1, "click_avatar": 1, "favorite": 1, "forward": 1,
                                "comment": 1, "follow": 1}
        self.estimator = None
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.stage = stage
        self.action = action
        tf.logging.set_verbosity(tf.logging.INFO)

    def build_estimator(self):
        if self.stage in ["evaluate", "offline_train"]:
            stage = "offline_train"
        else:
            stage = "online_train"
        model_checkpoint_stage_dir = os.path.join(FLAGS.model_checkpoint_dir, stage, self.action)
        if not os.path.exists(model_checkpoint_stage_dir):
            # 如果模型目录不存在，则创建该目录
            os.makedirs(model_checkpoint_stage_dir)
        elif self.stage in ["online_train", "offline_train"]:
            # 训练时如果模型目录已存在，则清空目录
            del_file(model_checkpoint_stage_dir)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.9, beta2=0.999,
                                           epsilon=1)
        config = tf.estimator.RunConfig(model_dir=model_checkpoint_stage_dir, tf_random_seed=SEED)
        self.estimator = tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_checkpoint_stage_dir,
            linear_feature_columns=self.linear_feature_columns,
            dnn_feature_columns=self.dnn_feature_columns,
            dnn_hidden_units=[32, 8],
            dnn_optimizer=optimizer,
            config=config)

    def df_to_dataset(self, df, stage, action, shuffle=True, batch_size=128, num_epochs=1):
        '''
        把DataFrame转为tensorflow dataset
        :param df: pandas dataframe. 
        :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
        :param action: String. Including "read_comment"/"like"/"click_avatar"/"favorite"/"forward"/"comment"/"follow"
        :param shuffle: Boolean. 
        :param batch_size: Int. Size of each batch
        :param num_epochs: Int. Epochs num
        :return: tf.data.Dataset object. 
        '''
        print(df.shape)
        print(df.columns)
        print("batch_size: ", batch_size)
        print("num_epochs: ", num_epochs)
        if stage != "submit":
            label = df[action]
            ds = tf.data.Dataset.from_tensor_slices((dict(df), label))
        else:
            ds = tf.data.Dataset.from_tensor_slices((dict(df)))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(df), seed=SEED)
        ds = ds.batch(batch_size)
        if stage in ["online_train", "offline_train"]:
            ds = ds.repeat(num_epochs)
        return ds

    def input_fn_train(self, df, stage, action, num_epochs):
        return self.df_to_dataset(df, stage, action, shuffle=True, batch_size=FLAGS.batch_size,
                                  num_epochs=num_epochs)

    def input_fn_predict(self, df, stage, action):
        return self.df_to_dataset(df, stage, action, shuffle=False, batch_size=len(df), num_epochs=1)

    def train(self):
        """
        训练单个行为的模型
        """
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action=self.action,
                                                                      day=STAGE_END_DAY[self.stage])
        stage_dir = os.path.join(FLAGS.root_path, self.stage, file_name)
        df = pd.read_csv(stage_dir)
        self.estimator.train(
            input_fn=lambda: self.input_fn_train(df, self.stage, self.action, self.num_epochs_dict[self.action])
        )

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
        evaluate_dir = os.path.join(FLAGS.root_path, self.stage, file_name)
        df = pd.read_csv(evaluate_dir)
        userid_list = df['userid'].astype(str).tolist()
        predicts = self.estimator.predict(
            input_fn=lambda: self.input_fn_predict(df, self.stage, self.action)
        )
        predicts_df = pd.DataFrame.from_dict(predicts)
        logits = predicts_df["logistic"].map(lambda x: x[0])
        labels = df[self.action].values
        uauc = uAUC(labels, logits, userid_list)
        return df[["userid", "feedid"]], logits, uauc

    
    def predict(self):
        '''
        预测单个行为的发生概率
        '''
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action="all",
                                                                      day=STAGE_END_DAY[self.stage])
        submit_dir = os.path.join(FLAGS.root_path, self.stage, file_name)
        df = pd.read_csv(submit_dir)
        t = time.time()
        predicts = self.estimator.predict(
            input_fn=lambda: self.input_fn_predict(df, self.stage, self.action)
        )
        predicts_df = pd.DataFrame.from_dict(predicts)
        logits = predicts_df["logistic"].map(lambda x: x[0])
        # 计算2000条样本平均预测耗时（毫秒）
        ts = (time.time()-t)*1000.0/len(df)*2000.0
        return df[["userid", "feedid"]], logits, ts

class Lightgbm(object):
    def __init__(self, stage, action):
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

    def pre_data(self, stage, action):
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
            # X = np.array(df.drop(ACTION_LIST, axis=1))
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
        
        x_train, self.x_test, y_train, self.y_test = train_test_split(X, y, test_size=0.10, random_state=100)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=100)
        self.train_data = lgb.Dataset(data=x_train,label=y_train)
        self.val_data = lgb.Dataset(data=x_val,label=y_val)

    def load_model(self):
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
            self.bst = lgb.Booster(model_file=os.path.join(self.model_checkpoint_stage_dir, 'model.txt'))

    def train(self):
        """
        训练单个行为的模型
        """
        param = {'num_leaves':31, 'num_trees':1, 'objective':'binary'}
        param['metric'] = 'binary_logloss'
        num_round = 10
        self.bst = lgb.train(param, self.train_data, num_round, valid_sets=[self.val_data])
        self.bst.save_model(os.path.join(self.model_checkpoint_stage_dir, 'model.txt'), num_iteration=self.bst.best_iteration)
        # self.bst = lgb.cv(param, train_data, num_round, nfold=5)

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

        ypred = self.bst.predict(self.x_test, num_iteration=self.bst.best_iteration)

        userid_list = df['userid'].astype(str).tolist()
        uauc = uAUC(self.y_test, ypred, userid_list)
        return ypred, uauc
    
    def predict(self):
        '''
        预测单个行为的发生概率
        '''
        t = time.time()
        logits = self.bst.predict(self.x_test, num_iteration=self.bst.best_iteration)
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
        model = Lightgbm(stage, action)
        model.pre_data(stage, action)
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
    # for action in ACTION_LIST:
    #     print("Action:", action)
    #     model = Lightgbm(linear_feature_columns, dnn_feature_columns, stage, action)
    #     model.build_estimator()

    #     if stage in ["online_train", "offline_train"]:
    #         # 训练 并评估
    #         model.train()
    #         ids, logits, action_uauc = model.evaluate()
    #         eval_dict[action] = action_uauc

    #     if stage == "evaluate":
    #         # 评估线下测试集结果，计算单个行为的uAUC值，并保存预测结果
    #         ids, logits, action_uauc = model.evaluate()
    #         eval_dict[action] = action_uauc
    #         predict_dict[action] = logits

    #     if stage == "submit":
    #         # 预测线上测试集结果，保存预测结果
    #         ids, logits, ts = model.predict()
    #         predict_time_cost[action] = ts
    #         predict_dict[action] = logits

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
    
