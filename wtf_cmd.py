
import yaml

from lib.models import *
from lib.DataPro import *
from wtf import *

timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wtf_log = os.path.join('../output/log', 'wtf_' + timestamp + '.log')
logging.basicConfig(filename=wtf_log, level=logging.DEBUG, filemode='w', format='%(funcName)s: %(message)s')

datapro_dict = {
    'DataPro': DataPro,
    'DataProLog': DataProLog,
    'DataProMedian': DataProMedian,
}

framework_dict = {
    'EncDec': EncDec,
    'Seq2Seq': Seq2Seq,
    'SparseAE': SparseAE,
    #'ContractiveAE': ContractiveAE,
}

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Need configuration file name.')
        exit()

    config_file = sys.argv[1]
    logging.info('Config file: %s' % config_file)
    config_file = './yaml/' + config_file
    with open(config_file, 'r') as f:
        pars = yaml.safe_load(f)

    data_pro = pars['DataProClass']
    framework = pars['Framework']
    logging.info('Framework: %s' % framework)
    load_len = pars['load_len']
    batch_size = pars['batch_size']

    DataProClass = datapro_dict[data_pro]
    Model = framework_dict[pars['Framework']]
    wtf = WTF(DataProClass, pars['DataPro'], Model, pars['Model'])
    if pars['command'] == 'run_train':
        wtf.run_train(epochs=pars['epochs'], kf=pars['kf'], save_freq=pars['save_freq'], base=wtf, config_file=config_file)
    elif pars['command'] == 'run_predict':
        wtf.run_predict(seq_len=pars['predict_seq_len'])

    '''
    if framework == 'EncDec':
        encoder = EncoderSimple(input_size=pars['encoder']['input_size'],
                                hidden_size=pars['encoder']['hidden_size'],
                                n_layers=pars['encoder']['n_layers'],
                                dropout=pars['encoder']['dropout'],
                                bidirectional=pars['encoder']['bidirectional']
                                )

        decoder = DecoderSimple(input_size=pars['decoder']['input_size'],
                                hidden_size=pars['decoder']['hidden_size'],
                                n_layers=pars['decoder']['n_layers'],
                                dropout=pars['decoder']['dropout'],
                                bidirectional=pars['decoder']['bidirectional']
                                )

        encdec = EncDec(encoder, decoder)

        Wtf = WTF(DataPro, pars['DataPro'])

        if pars['command'] == 'run_train':
            encdec.run_train(epochs=pars['epochs'], load_len=load_len)
    
    '''
