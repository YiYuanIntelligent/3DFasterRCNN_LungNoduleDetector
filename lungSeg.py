from preprocessing import full_prep
from config_submit import config as config_submit
import sys



datapath = config_submit[sys.argv[1]]
prep_result_path = config_submit[sys.argv[2]]


val = full_prep(datapath,prep_result_path,
                n_worker = config_submit['n_worker_preprocessing'],
                use_existing=config_submit['use_exsiting_preprocessing'])

