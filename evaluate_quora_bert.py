import numpy as np
import pandas as pd
import sklearn
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm

tqdm.pandas()

train_path = './train.csv'
test_path = './test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print('Train shape : ', train_df.shape)
print('Test shape : ', test_df.shape)

train_data = train_df[['question_text', 'target']]
train_data.columns = ['text', 'labels']
train_data, test_data = train_test_split(train_data, test_size=0.1, random_state=2020)

model_args = {
    "num_train_epochs": 5,
    "fp16": False,
    "overwrite_output_dir": True,
    "save_steps": 10000,
    "batch_size": 16
}


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


model_weight = 'checkpoint-12000'
model_path = f'outputs/{model_weight}'

model = ClassificationModel('bert', model_path, args=model_args, cuda_device=2)
# # model.train_model(train_data, acc=sklearn.metrics.f1_score)
#
result, model_outputs, wrong_predictions = model.eval_model(test_data, acc=sklearn.metrics.f1_score)

np.save(f'{model_weight}.npy', model_outputs)
# d = np.load(f'{model_weight}.npy')
pred_y = sigmoid(model_outputs)[:, 1]
print(result)
true_y = test_data['labels'].values
print(threshold_search(true_y, pred_y))
# print(d)
