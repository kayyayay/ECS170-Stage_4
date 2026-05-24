from code.stage_4_code.Dataset_Loader_Gen import Dataset_Loader_Joke
from code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_RNN_Gen import Setting_RNN_Gen
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy

import numpy as np
import torch

np.random.seed(2)
torch.manual_seed(2)

# Load the dataset
data_obj = Dataset_Loader_Joke('Jokes', '')
data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation/'

# Method
method_obj = Method_RNN_Generation(
    'RNN Generation',
    '',
    model_type='RNN'
)

# Saving the result
result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = '../../result/stage_4_result/'
result_obj.result_destination_file_name = 'Joke_generation_result'

# Setting
setting_obj = Setting_RNN_Gen('Generation setting', '')

# Run the model
evaluate_obj = Evaluate_Accuracy('evaluation', '')
setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)

setting_obj.print_setup_summary()

result = setting_obj.load_run_save_evaluate()

print("\nGENERATED JOKE")
print(result['sample'])

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

result = setting_obj.result.data

plt.plot(result['loss_list'])
plt.title(f"{method_obj.model_type} Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
