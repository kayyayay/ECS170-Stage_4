'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting


class Setting_RNN(setting):

    def load_run_save_evaluate(self):

        # -----------------------
        # 1. Load dataset
        # -----------------------
        loaded_data = self.dataset.load()

        # -----------------------
        # 2. Pass data to method (FIXED)
        # -----------------------
        self.method.data = loaded_data   # <-- IMPORTANT CHANGE

        # -----------------------
        # 3. Run model
        # -----------------------
        learned_result = self.method.run()

        # -----------------------
        # 4. Evaluate
        # -----------------------
        self.evaluate.data = {
            'true_y': learned_result['true_y'],
            'pred_y': learned_result['pred_y'],
        }
        metrics = self.evaluate.evaluate()

        # -----------------------
        # 5. Save results
        # -----------------------
        self.result.data = {
            'pred_y': learned_result['pred_y'],
            'true_y': learned_result['true_y'],
            'loss_list': learned_result['loss_list'],
            'acc_list': learned_result['acc_list'],
            'metrics': metrics,
        }
        self.result.save()

        return metrics