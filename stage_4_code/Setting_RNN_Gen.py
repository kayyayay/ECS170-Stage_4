'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting

class Setting_RNN_Gen(setting):

    def load_run_save_evaluate(self):

        data = self.dataset.load()

        self.method.build(data['vocab_size'])

        self.method.train_model(data['X_train'], data['y_train'])

        vocab = data['vocab']
        idx2word = {v: k for k, v in vocab.items()}

        sample = self.method.generate(
            "what did the",
            vocab,
            idx2word
        )

        print("\nGENERATED TEXT:\n", sample)

        self.result.data = {"sample": sample}
        self.result.save()

        return {"sample": sample}