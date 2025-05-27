from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import json
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_json(path):
    # 打开并加载JSON文件
    with open(path, 'r') as f:
        data = json.load(f)
    return data

'''def train_and_test(X, y):

    # 切分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10,stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
        # 创建朴素贝叶斯分类器
    clf = LogisticRegression(multi_class='multinomial',penalty='none')

    # 训练模型
    clf.fit(X_train_s, y_train)

    # 预测测试集
    predictions = clf.predict(X_test_s)
    # 计算并打印评估指标
    print('Accuracy:', metrics.accuracy_score(y_test, predictions))
    print('Precision:', metrics.precision_score(y_test, predictions))
    print('Recall:', metrics.recall_score(y_test, predictions))
    print('F1 Score:', metrics.f1_score(y_test, predictions))'''


from sklearn.model_selection import cross_val_score
from scipy.stats import ttest_rel

def train_and_test(X, y):
    # 标准化整个数据集
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    # 创建逻辑回归分类器
    clf = LogisticRegression(multi_class='multinomial',penalty='none')

    
    # 使用十折交叉验证计算每个评价指标的分数，并存储在列表中
    accuracy_scores = cross_val_score(clf, X_s, y, cv=10, scoring='accuracy').tolist()
    precision_scores = cross_val_score(clf, X_s, y, cv=10, scoring='precision_macro').tolist()
    recall_scores = cross_val_score(clf, X_s, y, cv=10, scoring='recall_macro').tolist()
    f1_scores = cross_val_score(clf, X_s, y, cv=10, scoring='f1_macro').tolist()
    '''accuracy_scores.remove(max(accuracy_scores))
    precision_scores.remove(max(precision_scores))
    recall_scores.remove(max(recall_scores))
    f1_scores.remove(max(f1_scores))
    accuracy_scores.remove(min(accuracy_scores))
    precision_scores.remove(min(precision_scores))
    recall_scores.remove(min(recall_scores))
    f1_scores.remove(min(f1_scores))'''
    # 输出每个指标的十折交叉验证分数集合
    '''print('Accuracy=', accuracy_scores)
    print('Precision=', precision_scores)
    print('Recall=', recall_scores)
    print('F1=', f1_scores)'''
    
    
    '''print('accuracy差值：',max(accuracy_scores)-np.mean(accuracy_scores))
    print('precision差值：',max(precision_scores)-np.mean(precision_scores))
    print('recall差值：',max(recall_scores)-np.mean(recall_scores))
    print('f1差值：',max(f1_scores)-np.mean(f1_scores))'''
    
    print(f'{np.mean(accuracy_scores)*100:.2f}±{np.std(accuracy_scores)*100:.2f}')
    print(f'{np.mean(precision_scores)*100:.2f}±{np.std(precision_scores)*100:.2f}')
    print(f'{np.mean(recall_scores)*100:.2f}±{np.std(recall_scores)*100:.2f}')
    print(f'{np.mean(f1_scores)*100:.2f}±{np.std(f1_scores)*100:.2f}')

    return accuracy_scores,precision_scores,recall_scores,f1_scores



bug_lists = ['access_control','arithmetic','denial_of_service','front_running','reentrancy','time_manipulation','unchecked_low_level_calls']
for bug_type in bug_lists:
    save_path = f'logs/graph_classification/source_code/metapath2vec/ast/{bug_type}/ast_cfg.json'
    data = load_json(save_path)
    all_vector_lists = []
    label_lists = []
    for object in data:
        vector = object['fusion']
        label = object['label']
        all_vector_lists.append(vector)
        label_lists.append(label)

    if(bug_type=='access_control'):
        accuracy_scores,precision_scores,recall_scores,f1_scores = train_and_test(np.array(all_vector_lists),np.array(label_lists))
        '''t_stat, p_value_Accuracy = ttest_rel(accuracy_scores,AC_Accuracy)
        t_stat, p_value_Precision = ttest_rel(precision_scores,AC_Precision)
        t_stat, p_value_Recall = ttest_rel(recall_scores,AC_Recall)
        t_stat, p_value_F1 = ttest_rel(f1_scores,AC_F1)
        print(p_value_Accuracy)
        print(p_value_Precision)
        print(p_value_Recall)
        print(p_value_F1)'''
    
    
    if(bug_type=='arithmetic'):
        
        accuracy_scores,precision_scores,recall_scores,f1_scores = train_and_test(np.array(all_vector_lists),np.array(label_lists))
        '''t_stat, p_value_Accuracy = ttest_rel(accuracy_scores,AR_Accuracy)
        t_stat, p_value_Precision = ttest_rel(precision_scores,AR_Precision)
        t_stat, p_value_Recall = ttest_rel(recall_scores,AR_Recall)
        t_stat, p_value_F1 = ttest_rel(f1_scores,AR_F1)
        print(p_value_Accuracy)
        print(p_value_Precision)
        print(p_value_Recall)
        print(p_value_F1)'''
    if(bug_type=='denial_of_service'):
        
        accuracy_scores,precision_scores,recall_scores,f1_scores = train_and_test(np.array(all_vector_lists),np.array(label_lists))
        '''t_stat, p_value_Accuracy = ttest_rel(accuracy_scores,DS_Accuracy)
        t_stat, p_value_Precision = ttest_rel(precision_scores,DS_Precision)
        t_stat, p_value_Recall = ttest_rel(recall_scores,DS_Recall)
        t_stat, p_value_F1 = ttest_rel(f1_scores,DS_F1)
        print(p_value_Accuracy)
        print(p_value_Precision)
        print(p_value_Recall)
        print(p_value_F1)'''
    if(bug_type=='front_running'):
        
        accuracy_scores,precision_scores,recall_scores,f1_scores = train_and_test(np.array(all_vector_lists),np.array(label_lists))
        
        '''t_stat, p_value_Accuracy = ttest_rel(accuracy_scores,FR_Accuracy)
        t_stat, p_value_Precision = ttest_rel(precision_scores,FR_Precision)
        t_stat, p_value_Recall = ttest_rel(recall_scores,FR_Recall)
        t_stat, p_value_F1 = ttest_rel(f1_scores,FR_F1)
        print(p_value_Accuracy)
        print(p_value_Precision)
        print(p_value_Recall)
        print(p_value_F1)'''
    if(bug_type=='reentrancy'):
        
        accuracy_scores,precision_scores,recall_scores,f1_scores = train_and_test(np.array(all_vector_lists),np.array(label_lists))
        '''t_stat, p_value_Accuracy = ttest_rel(accuracy_scores,RE_Accuracy)
        t_stat, p_value_Precision = ttest_rel(precision_scores,RE_Precision)
        t_stat, p_value_Recall = ttest_rel(recall_scores,RE_Recall)
        t_stat, p_value_F1 = ttest_rel(f1_scores,RE_F1)
        print(p_value_Accuracy)
        print(p_value_Precision)
        print(p_value_Recall)
        print(p_value_F1)'''
    if(bug_type=='time_manipulation'):
        
        accuracy_scores,precision_scores,recall_scores,f1_scores = train_and_test(np.array(all_vector_lists),np.array(label_lists))
        '''t_stat, p_value_Accuracy = ttest_rel(accuracy_scores,TM_Accuracy)
        t_stat, p_value_Precision = ttest_rel(precision_scores,TM_Precision)
        t_stat, p_value_Recall = ttest_rel(recall_scores,TM_Recall)
        t_stat, p_value_F1 = ttest_rel(f1_scores,TM_F1)
        print(p_value_Accuracy)
        print(p_value_Precision)
        print(p_value_Recall)
        print(p_value_F1)'''
    if(bug_type=='unchecked_low_level_calls'):
        
        accuracy_scores,precision_scores,recall_scores,f1_scores = train_and_test(np.array(all_vector_lists),np.array(label_lists))
        '''t_stat, p_value_Accuracy = ttest_rel(accuracy_scores,UC_Accuracy)
        t_stat, p_value_Precision = ttest_rel(precision_scores,UC_Precision)
        t_stat, p_value_Recall = ttest_rel(recall_scores,UC_Recall)
        t_stat, p_value_F1 = ttest_rel(f1_scores,UC_F1)
        print(p_value_Accuracy)
        print(p_value_Precision)
        print(p_value_Recall)
        print(p_value_F1)'''
    
        


