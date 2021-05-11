# **2021中国高校计算机大赛-微信大数据挑战赛Baseline**

本次比赛基于脱敏和采样后的数据信息，对于给定的一定数量到访过微信视频号“热门推荐”的用户，根据这些用户在视频号内的历史n天的行为数据，通过算法在测试集上预测出这些用户对于不同视频内容的互动行为（包括点赞、点击头像、收藏、转发等）的发生概率。 

本次比赛以多个行为预测结果的加权uAUC值进行评分。大赛官方网站：https://algo.weixin.qq.com/

## **1. 环境配置**

- pandas>=1.0.5
- tensorflow>=1.14.0
- python3

## **2. 运行配置**

- CPU/GPU均可
- 最小内存要求
    - 特征/样本生成：3G
    - 模型训练及评估：6G

- 耗时
    - 测试环境：内存8G，CPU 2.3 GHz 双核Intel Core i5
    - 特征/样本生成：226 s
    - 模型训练及评估：740 s 
    
## **3. 目录结构**

- comm.py: 数据集生成
- baseline.py: 模型训练，评估，提交
- evaluation.py: uauc 评估
- data/: 数据，特征，模型
    - wechat_algo_data1/: 初赛数据集
    - feature/: 特征
    - offline_train/：离线训练数据集
    - online_train/：在线训练数据集
    - evaluate/：评估数据集
    - submit/：在线预估结果提交
    - model/: 模型文件

## **4. 运行流程**
- 新建data目录，下载比赛数据集，放在data目录下并解压，得到wechat_algo_data1目录
- 生成特征/样本：python comm.py （自动新建data目录下用于存储特征、样本和模型的各个目录）
- 训练离线模型：python baseline.py offline_train 
- 评估离线模型：python baseline.py evaluate  （生成data/evaluate/submit_${timestamp}.csv）
- 训练在线模型：python baseline.py online_train 
- 生成提交文件：python baseline.py submit  （生成data/submit/submit_${timestamp}.csv）
- 评估代码: evaluation.py

## **5. 模型及特征**
- 模型：[Wide & Deep](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454)
- 参数：
    - batch_size: 128
    - emded_dim: 10
    - num_epochs: 1
    - learning_rate: 0.1
- 特征：
    - dnn 特征: userid, feedid, authorid, bgm_singer_id, bgm_song_id
    - linear 特征：videoplayseconds, device，用户/feed 历史行为次数
  
## **6. 模型结果**

|stage  |weight_uauc |read_comment|like|click_avatar|forward| 
|:---- |:----  |:----  |:----  |:----  |:----|
| 离线  | 0.657003 |0.626822 |0.633864  |0.735366 |0.690416 | 
| 在线  | 0.607908| 0.577496 |0.588645  |0.682383  |0.638398 | 
   
## **7. 相关文献**
* Cheng, Heng-Tze, et al. "Wide & deep learning for recommender systems." Proceedings of the 1st workshop on deep learning for recommender systems. 2016.

   



