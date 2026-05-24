from code.stage_4_code.Dataset_Loader_Clas import Dataset_Loader_Text
from code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_RNN_Clas import Setting_RNN
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy

import numpy as np
import torch

np.random.seed(2)
torch.manual_seed(2)

# The dataset
data_obj = Dataset_Loader_Text('IMDB', '')
data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification/'
data_obj.dataset_name = 'IMDB'

# Method
method_obj = Method_RNN_Classification(
    'RNN Classification',
    '',
    model_type='RNN'
)

# Saving the result
result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = '../../result/stage_4_result/'
result_obj.result_destination_file_name = 'IMDB_prediction_result'

# Setting
setting_obj = Setting_RNN('RNN setting', '')
evaluate_obj = Evaluate_Accuracy('accuracy', '')

# Run
setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
setting_obj.print_setup_summary()
metrics = setting_obj.load_run_save_evaluate()

print('IMDB RNN Results')
print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1:        {metrics['f1']:.4f}")

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

result = setting_obj.result.data

plt.plot(result['loss_list'])
plt.title(f"{method_obj.model_type} Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(result['acc_list'])
plt.title(f"{method_obj.model_type} Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
