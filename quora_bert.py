import pandas as pd
import sklearn
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import accuracy_score
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
    "save_steps": 2000,
    "train_batch_size": 16,
    "eval_batch_size": 8,
    "use_early_stopping": True,
    "early_stopping_delta": 0.01,
    "early_stopping_metric": "acc",
    "early_stopping_metric_minimize": False,
    "early_stopping_patience": 5,
    "evaluate_during_training_steps": 1000,
    "learning_rate": 1e-4,
    "wandb_project": "bert-quora"
}

model = ClassificationModel('bert', 'bert-base-cased', args=model_args, cuda_device=1)
model.train_model(train_data, eval_df=test_data, acc=sklearn.metrics.f1_score)

result, model_outputs, wrong_predictions = model.eval_model(test_data, acc=sklearn.metrics.f1_score)

print(result)
