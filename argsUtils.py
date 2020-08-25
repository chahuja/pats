import argparse
import itertools
from ast import literal_eval

def get_args_perm():
  parser = argparse.ArgumentParser()

  ## Dataset Parameters
  parser.add_argument('-path2data', nargs='+', type=str, default=['pats/data/'],
                      help='path to data')
  parser.add_argument('-speaker', nargs='+', type=literal_eval, default=['bee'],
                      help='choose speaker or `all` to use all the speakers available')  
  parser.add_argument('-modalities', nargs='+', type=literal_eval, default=[['pose/data', 'audio/log_mel_512']],
                      help='choose a set of modalities to be loaded by the dataloader')  
  parser.add_argument('-split', nargs='+', type=literal_eval, default=[None],
                      help='(train,dev) split of data. default=None')
  parser.add_argument('-batch_size', nargs='+', type=int, default=[32],
                      help='minibatch size. Use batch_size=1 when using time=0')
  parser.add_argument('-shuffle', nargs='+', type=int, default=[1],
                      help='shuffle the data after each epoch. default=True')
  parser.add_argument('-time', nargs='+', type=int, default=[4.3],
                      help='time (in seconds) for each sample')
  parser.add_argument('-fs_new', nargs='+', type=literal_eval, default=[[15, 15]],
                      help='subsample to the new frequency')  
  
  
  args, unknown = parser.parse_known_args()
  print(args)
  print(unknown)

  ## Create a permutation of all the values in argparse
  args_dict = args.__dict__
  args_keys = sorted(args_dict)
  args_perm = [dict(zip(args_keys, prod)) for prod in itertools.product(*(args_dict[names] for names in args_keys))]
  
  return args, args_perm

def argparseNloop(loop):
  args, args_perm = get_args_perm()

  for i, perm in enumerate(args_perm):
    args.__dict__.update(perm)
    print(args)    
    loop(args, i)
