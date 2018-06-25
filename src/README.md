# Few Shot Music Generation
TODO(all): Add more documentation.

## Instructions
Download the zip and csv files from
https://drive.google.com/corp/drive/u/0/folders/1sI1K3CjzpN81QjjpaEDVKW79c7AOUdyQ
and store them in `../raw-data/lyrics` and `../raw-data/freemidi`, respectively.

Unzip the zip files.

Sample run (check the different yaml files for different ways to run):
```
$ CONFIG=lyrics.yaml
$ MODEL=lstm_baseline.yaml
$ TASK=5shot.yaml
python -um train.train --data=config/${CONFIG} --model=config/${MODEL} --task=config/${TASK} --checkpt_dir=/tmp/fewshot/lstm_baseline
```

To view the tensorboard (only works for `lstm_baseline.yaml` model):
```
$ tensorboard --logdir=/tmp/fewshot
```
