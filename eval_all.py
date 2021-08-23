import datetime
import yaml

with open('eval_log.txt', 'w') as f:
    f.write(str(datetime.datetime.now()))

sample = 'val'
tasks = ['id', 'ood_ads', 'ood_cat', 'ood_both']

for task in tasks:
    sample_filename = '../ocp_datasets/data/is2re/tiny/{}/data.lmdb'.format(sample+'_'+task)
    with open('configs/is2re/10k/base.yml') as f:
        #read config
        base_yml = yaml.load(f, Loader=yaml.FullLoader)
        #change it
        base_yml['dataset'][1]['src'] = sample_filename
        base_yml['dataset'][2]['src'] = sample_filename
    #write new config
    with open('configs/is2re/10k/base.yml', 'w') as f:
        yaml.dump(base_yml, f)
    #launch main.py
    exec(open('main.py').read())



