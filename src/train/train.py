import os
import sys
import pprint
import argparse
import yaml
from tqdm import tqdm
from importlib import import_module

from data.episode import load_sampler_from_config

PP = pprint.PrettyPrinter(depth=6)
LOG_FILE = 'status.log'


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


def write_samples(model, episode_sampler, samples_dir, n_samples, max_len):
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    for i in range(n_samples):
        curr_sample_dir = os.path.join(samples_dir, 'sample_%d' % i)
        os.makedirs(curr_sample_dir)

        episode = episode_sampler.get_episode()
        support_set = episode.support[0]
        sample = model.sample(support_set, max_len)

        for j in range(support_set.shape[0]):
            write_seq(episode_sampler.detokenize(support_set[j]),
                      curr_sample_dir, 'support_%d' % j)

        write_seq(episode_sampler.detokenize(sample), curr_sample_dir,
                  'model_sample')


def create_log(checkpt_dir):
    if checkpt_dir != '':
        if not os.path.exists(checkpt_dir):
            os.makedirs(checkpt_dir)

        sys.stdout = open(os.path.join(checkpt_dir, LOG_FILE), 'w', 0)


parser = argparse.ArgumentParser(description='Train a model.')
parser.add_argument('--data', dest='data', default='')
parser.add_argument('--model', dest='model', default='')
parser.add_argument('--task', dest='task', default='')
parser.add_argument('--checkpt_dir', dest='checkpt_dir', default='')
parser.add_argument('--init_dir', dest='init_dir', default='')
parser.add_argument('--mode', dest='mode', default='train')
args = parser.parse_args()


def main():
    create_log(args.checkpt_dir)
    print('Args:')
    print(PP.pformat(vars(args)))

    config = yaml.load(open(args.data, 'r'))
    config.update(yaml.load(open(args.task, 'r')))
    config.update(yaml.load(open(args.model, 'r')))
    config['dataset_path'] = os.path.abspath(config['dataset_path'])
    config['checkpt_dir'] = args.checkpt_dir
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

    save_best_val = False
    if 'patience_iters' in config:
        save_best_val = True
        patience_iters = config['patience_iters']

    model = load_model_from_config(config)
    model.recover_or_init(args.init_dir)

    if args.mode == 'train':
        # Train model and evaluate
        avg_nll = evaluate(model, episode_sampler['val'], n_val)
        print("Iter: %d, val-nll: %.3e" % (0, avg_nll))

        avg_loss = 0.
        best_val_nll = sys.float_info.max
        for i in tqdm(range(1, n_train + 1)):
            episode = episode_sampler['train'].get_episode()
            loss = model.train(episode)
            avg_loss += loss

            if i % print_every_n == 0:
                print("Iter: %d, loss: %.3e" % (i, avg_loss / print_every_n))
                avg_loss = 0.

            if i % val_every_n == 0:
                avg_nll = evaluate(model, episode_sampler['val'], n_val)
                print("Iter: %d, val-nll: %.3e" % (i, avg_nll))

                if save_best_val:
                    if avg_nll < best_val_nll:
                        best_val_nll = avg_nll
                        print("=> Found winner on validation set: %.3e" % best_val_nll)
                        if args.checkpt_dir != '':
                            model.save(args.checkpt_dir)

                        patience_iters = min(
                            config['patience_iters'], patience_iters + 1)
                    else:
                        patience_iters -= 1
                        print("=> Decreasing patience: %d" % patience_iters)
                        if patience_iters == 0:
                            print("=> Patience exhausted - ending training...")
                            break
                else:
                    if args.checkpt_dir != '':
                        model.save(args.checkpt_dir)

        # Load best model
        if args.checkpt_dir != '':
            print("=> Loading winner on validation set")
            model.recover_or_init(args.checkpt_dir)

    # Evaluate model after training on training, validation, and test sets
    avg_nll = evaluate(model, episode_sampler['train'], n_test)
    print("Train Avg NLL: %.3e" % (avg_nll))
    avg_nll = evaluate(model, episode_sampler['val'], n_test)
    print("Validation Avg NLL: %.3e" % (avg_nll))
    avg_nll = evaluate(model, episode_sampler['test'], n_test)
    print("Test Avg NLL: %.3e" % (avg_nll))

    # Generate samples from trained model for test episodes
    if args.checkpt_dir != '':
        samples_dir = os.path.join(args.checkpt_dir, 'samples')
        write_samples(
            model, episode_sampler['test'], samples_dir, n_samples, max_len)


if __name__ == '__main__':
    main()
