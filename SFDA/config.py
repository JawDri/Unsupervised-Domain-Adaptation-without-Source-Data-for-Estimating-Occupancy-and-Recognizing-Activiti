import os
import yaml
import easydict
from os.path import join


class Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)


import argparse
parser = argparse.ArgumentParser(description='Code for *Learning to Transfer Examples for Partial Domain Adaptation*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')

args = parser.parse_args()

config_file = args.config

args = yaml.full_load(open(config_file))

save_config = yaml.full_load(open(config_file))

args = easydict.EasyDict(args)

dataset = None
if args.data.dataset.name == 'office':
    dataset = Dataset(
    path=args.data.dataset.root_path,
    domains=['Source', 'Target'],
    files=[
        'Source_train.csv',
        'Target_train.csv',
        'Source_test.csv',
        'Target_test.csv'
    ],
    prefix=args.data.dataset.root_path)
else:
    raise Exception(f'dataset {args.data.dataset.name} not supported!')
# print (args.data.dataset.source)
source_domain_name = dataset.domains[args.data.dataset.source]
target_domain_name = dataset.domains[args.data.dataset.target]
source_file = dataset.files[args.data.dataset.source]
target_file = dataset.files[args.data.dataset.target]
print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
print ("source_domain_name :", source_domain_name, "target_domain_name :", target_domain_name)
