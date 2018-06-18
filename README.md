# Few-Shot Music Generation

## Few-Shot Distribution Learning for Music Generation

* Tagline: Learning a generative model for music data using a small amount of examples.
* Date: December 2017
* Category: Fundamental Research
* Author(s): [Hugo Larochelle](https://github.com/larocheh), [Chelsea Finn](https://github.com/cbfinn), [Sachin Ravi](https://github.com/sachinravi14)

## Project Status

* ~~Brainstorming for datasets phase: Currently collecting ideas for dataset collection for lyrics and MIDI data. See the Issues page for details.~~
* ~~Collecting actual data for lyrics and MIDI.~~
* ~~Decide and implement pre-processing scheme for data (specifically for MIDI).~~
* Release training script and model API code.

## Community Links

* [Project Slack](https://few-shot-music-gen.slack.com/join/shared_invite/enQtMjgwMTA0NTA3MzQ3LTA3MTc3M2E4MjEyNDlhZDNlMTU2ZmUyMmNmMDlhYmQ2ZmFkMDRiZTAzZDJmYmYwYmE0NjRmZGMyMmYxOWEzMWU)
* [Project Mailing List](https://groups.google.com/forum/#!forum/few-shot-music-generation)

## Problem description:

See Introduction section of the proposal.

## Why this problem matters:

See Introduction section of the proposal.

## How to measure success:

See Experiments section of the proposal.

## Datasets:

See Datasets subsection of the proposal.

## Relevant Work:

See Related Work section of the proposal.

## Contribute

* Please begin by reading papers from the [Reading List](https://github.com/AI-ON/Few-Shot-Music-Generation/blob/master/READING_LIST.md) to familiarize yourself with work in this area.

### Data

Both the lyrics and freemidi data can be downloaded [here](https://drive.google.com/drive/u/1/folders/1sI1K3CjzpN81QjjpaEDVKW79c7AOUdyQ). Place the `raw-data` directory in the home folder of the repository and make sure to unzip both `.zip` files in the data sub-directories.

For example, for the lyrics data, make sure in the given path the following files and directories exist:
```
ls Few-Shot-Music-Generation/raw-data/lyrics/
>> lyrics_data test.csv  train.csv  val.csv
```

### Training Models
The training script can be run with the following command:

```
python -um train.train --model=config/[model yaml file] --data=config/[data yaml file] --task=config/[task yaml file] --checkpt_dir=[snapshot directory for model]
```

If you have any trouble running the code, please create an issue describing your problem.
