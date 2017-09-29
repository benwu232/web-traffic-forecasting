import sklearn

from lib.foundation import *
from lib.DataPro import *

#main class of Wether Traffic Forecast
class WTF(object):
    def __init__(self, DataPro, data_pro_pars, Model, model_pars):
        #init data
        self.init_data(DataPro, data_pro_pars)
        #load model
        self.init_model(Model, model_pars, self.data_pro)

    def init_data(self, DataPro, data_pro_pars):
        self.data_pro = DataPro(data_pro_pars)
        self.data_pro.sel_batch_builder()

    def init_model(self, Model=None, model_pars={}, DataPro=None):
        self.model = Model(model_pars)
        self.model.transform = DataPro.torch_transform
        self.model.inv_transform = DataPro.torch_inv_transform

    def run_train(self, epochs, **kwargs):
        kf = 0
        if 'kf' in kwargs:
            kf = kwargs['kf']

        self.data_pro.sel_batch_builder(kf)
        self.model.run_train(self.data_pro.train_bb, self.data_pro.validate_bb, epochs=epochs, **kwargs)

    def run_predict(self, seq_len=None):
        print('Preparing data ...')
        self.data_pro.predict_batch_builder()
        #self.model.load_model(model_file[0], model_file[1])
        print('Predicting ...')
        predict_results = self.model.predict(self.data_pro.predict_bb, seq_len)
        #predict_results = self.data_pro.np_inv_transform(predict_results)
        #No 9.12 data
        predict_results = predict_results[:, 1:]

        print('Generating submission file...')
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        submission_file = 'submission_' + timestamp + '.zip'
        gen_submission(self.data_pro.page_info, predict_results, submission_file, key_file=self.data_pro.key_file)
        print('Bingo!')
