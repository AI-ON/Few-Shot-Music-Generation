import os
import pprint
import argparse
import yaml

from data.episode import load_sampler_from_config
from .train import load_model_from_config

PP = pprint.PrettyPrinter(depth=6)

parser = argparse.ArgumentParser(description='Train a model.')
parser.add_argument('--data', dest='data', default='')
parser.add_argument('--task', dest='task', default='')
args = parser.parse_args()


def main():
    print('Args:')
    print(PP.pformat(vars(args)))

    config = yaml.load(open(args.data, 'r'))
    config.update(yaml.load(open(args.task, 'r')))
    config.update(yaml.load(open('config/lstm_baseline_test_seed.yaml', 'r')))
    config['dataset_path'] = os.path.abspath(config['dataset_path'])
    print('Config:')
    print(PP.pformat(config))

    episode_sampler = {}
    config['split'] = 'train'
    episode_sampler['train'] = load_sampler_from_config(config)

    config['input_size'] = episode_sampler['train'].get_num_unique_words()
    if not config['input_size'] > 0:
        raise RuntimeError(
            'error reading data: %d unique tokens processed' % config['input_size'])
    print('Num unique words: %d' % config['input_size'])

    model = load_model_from_config(config)
    model.recover_or_init('')

    ############################################################################
    # Run test to compare loss on training set after n updates to what we expect
    ############################################################################

    EXP_LOSS = {
        'config/midi.yaml' + 'config/5shot.yaml': 6.1458707,
        'config/lyrics.yaml' + 'config/5shot.yaml': 7.1389594
    }
    EPSILON = 0.001
    N_UPDATES = 10
    ERROR_MSG = """ Test failed: there is an issue with the seeding as model
                    loss is different from what we expect """

    loss = 0.
    for i in range(0, N_UPDATES):
        episode = episode_sampler['train'].get_episode()
        loss = model.train(episode)

    k = args.data + args.task
    if k not in EXP_LOSS:
        raise RuntimeError(
            'No test for data: %s and task: %s' % (args.data, args.task))

    expected_loss = EXP_LOSS[k]
    assert abs(loss - expected_loss) <= EPSILON, ERROR_MSG


if __name__ == '__main__':
    main()
