import os
import pprint
import argparse
import yaml
from importlib import import_module

from data.episode import load_sampler_from_config

PP = pprint.PrettyPrinter(depth=6)


def load_model_from_config(config):
    Model = getattr(import_module(config['model_module_name']),
                    config['model_class_name'])
    return Model(config)


def write_seq(seq, dir, name):
    if isinstance(seq, str):
        text_file = open(os.path.join(dir, name + '.txt'), "w")
        text_file.write(seq)
        text_file.close()
    else:
        seq.write(os.path.join(dir, name + '.mid'))


def evaluate(model, episode_sampler, n_episodes):
    avg_nll = 0.
    for i in range(n_episodes):
        episode = episode_sampler.get_episode()
        avg_nll += model.eval(episode)

    return avg_nll / n_episodes


parser = argparse.ArgumentParser(description='Train a model.')
parser.add_argument('--data', dest='data', default='')
parser.add_argument('--model', dest='model', default='')
parser.add_argument('--task', dest='task', default='')
parser.add_argument('--checkpt_dir', dest='checkpt_dir', default='')
parser.add_argument('--init_dir', dest='init_dir', default='')
args = parser.parse_args()
print('Args:')
print(PP.pformat(vars(args)))

config = yaml.load(open(args.data, 'r'))
config.update(yaml.load(open(args.task, 'r')))
config.update(yaml.load(open(args.model, 'r')))
config['dataset_path'] = os.path.abspath(config['dataset_path'])
print('Config:')
print(PP.pformat(config))


episode_sampler = {}
for split in config['splits']:
    config['split'] = split
    episode_sampler[split] = load_sampler_from_config(config)

config['input_size'] = episode_sampler['train'].get_num_unique_words()
if not config['input_size'] > 0:
    raise RuntimeError(
        'error reading data: %d unique tokens processed' % config['input_size'])
print('Num unique words: %d' % config['input_size'])

n_train = config['n_train']
print_every_n = config['print_every_n']
val_every_n = config['val_every_n']
n_val = config['n_val']
n_test = config['n_test']
n_samples = config['n_samples']
max_len = config['max_len']

model = load_model_from_config(config)
model.recover_or_init(args.init_dir)

# Train model and evaluate
avg_nll = evaluate(model, episode_sampler['val'], n_val)
print("Iter: %d, val-nll: %.3e" % (0, avg_nll))

avg_loss = 0.
for i in range(1, n_train + 1):
    episode = episode_sampler['train'].get_episode()
    loss = model.train(episode)
    avg_loss += loss

    if i % val_every_n == 0:
        avg_nll = evaluate(model, episode_sampler['val'], n_val)
        print("Iter: %d, val-nll: %.3e" % (i, avg_nll))

        if args.checkpt_dir != '':
            model.save(args.checkpt_dir)

    if i % print_every_n == 0:
        print("Iter: %d, loss: %.3e" % (i, avg_loss / print_every_n))
        avg_loss = 0.

# Evaluate model after training on training, validation, and test sets
avg_nll = evaluate(model, episode_sampler['train'], n_val)
print("Train Avg NLL: %.3e" % (avg_nll))
avg_nll = evaluate(model, episode_sampler['val'], n_val)
print("Validation Avg NLL: %.3e" % (avg_nll))
avg_nll = evaluate(model, episode_sampler['test'], n_val)
print("Test Avg NLL: %.3e" % (avg_nll))

# Generate samples from trained model for test episodes
samples_dir = os.path.join(args.checkpt_dir, 'samples')
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

for i in range(n_samples):
    curr_sample_dir = os.path.join(samples_dir, 'sample_%d' % i)
    os.makedirs(curr_sample_dir)

    episode = episode_sampler['test'].get_episode()
    support_set = episode.support[0]
    sample = model.sample(support_set, max_len)

    for j in range(support_set.shape[0]):
        write_seq(episode_sampler['test'].detokenize(support_set[j]),
                  curr_sample_dir, 'support_%d' % j)

    write_seq(episode_sampler['test'].detokenize(sample), curr_sample_dir,
              'model_sample')
