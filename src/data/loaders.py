#!/usr/bin/python3
"""A module for lyrics and MIDI dataset loaders.

There are two loaders: MIDILoader and LyricsLoader. These loaders are
essentially file readers which additionally turn the data into a series of
integer tokens.
"""
import os
import numpy as np
import logging
import nltk
import string
import collections
import math
from operator import itemgetter
import pretty_midi

_SUSTAIN_ON = 0
_SUSTAIN_OFF = 1
_NOTE_ON = 2
_NOTE_OFF = 3

NOTE_ON = 1
NOTE_OFF = 2
TIME_SHIFT = 3
VELOCITY = 4

MAX_SHIFT_STEPS = 100

MIN_MIDI_VELOCITY = 1
MAX_MIDI_VELOCITY = 127

MIN_MIDI_PROGRAM = 0
MAX_MIDI_PROGRAM = 127
PROGRAMS_PER_FAMILY = 8


log = logging.getLogger("few-shot")


class Loader(object):
    """A class for turning data into a sequence of tokens.
    """
    def __init__(self, max_len, dtype=np.int32, persist=True):
        self.max_len = max_len
        self.dtype = dtype
        self.persist = persist

    def is_song(self, filepath):
        raise NotImplementedError

    def read(self, filepath):
        raise NotImplementedError

    def tokenize(self, data):
        raise NotImplementedError

    def detokenize(self, numpy_data):
        raise NotImplementedError

    def validate(self, filepath):
        try:
            self.load(filepath)
            return True
        except OSError:
            return False
        except KeyError:
            return False
        except EOFError:
            return False
        except IndexError:
            return False
        except ValueError:
            return False

    def load(self, filepath):
        npfile = '%s.%s.npy' % (filepath, self.max_len)
        if self.persist and os.path.isfile(npfile):
            return np.load(npfile).astype(self.dtype)
        else:
            data = self.read(filepath)
            tokens = self.tokenize(data)
            numpy_tokens = np.zeros(self.max_len, dtype=self.dtype)
            for token_index in range(min(self.max_len, len(tokens))):
                numpy_tokens[token_index] = tokens[token_index]
            if self.persist:
                np.save(npfile, numpy_tokens)
            return numpy_tokens


class MIDILoader(Loader):
    """Objects of this class parse MIDI files into a sequence of note IDs
    """
    def read(self, filepath):
        """Reads a MIDI file.

        Arguments:
            filepath (str): path to the lyrics file. e.g.
                "/home/user/freemidi_data/Tool/lateralus.mid"

        """
        return pretty_midi.PrettyMIDI(filepath)

    def is_song(self, filepath):
        return filepath.endswith('.mid')

    def tokenize(self, midi):
        """Turns a MIDI file into a list of event IDs.

        Arguments:
            filepath (str): path to the lyrics file. e.g.
                "/home/user/freemidi_data/Tool/lateralus.mid"
        """
        tokens = []
        midi_notes = get_notes(midi)
        midi_control_changes = get_control_changes(midi)
        midi_notes, midi_control_changes = apply_sustain_control_changes(midi_notes,
            midi_control_changes)
        midi_notes = quantize_notes(midi_notes)
        events = get_event_list(midi_notes)
        for event_type, event_value, family in events:
            if event_type == NOTE_ON:
                token = family * 128 + event_value
            elif event_type == NOTE_OFF:
                token = 16 * 128 + family * 128 + event_value
            elif event_type == VELOCITY:
                token = 16 * 128 * 2 + 32 * family + event_value
            elif event_type == TIME_SHIFT:
                token = 16 * 128 * 2 + 32 * 16 + event_value
            tokens.append(token)
        return tokens

    def detokenize(self, numpy_data):
        current_time = 0
        current_velocity = [64 for _ in range(16)]
        unsorted_notes = [[] for _ in range(16)]
        active_notes = [[None for _ in range(128)] for _ in range(16)]
        for token in numpy_data:
            if token < 16 * 128:
                instr_class = token // 128
                note_number = token % 128
                active_notes[instr_class][note_number] = (current_velocity[instr_class], current_time)
            elif token < 16 * 128 * 2:
                instr_class = (token-16*128) // 128
                pitch = (token-16*128) % 128
                (velocity, start_time) = active_notes[instr_class][pitch]
                unsorted_notes[instr_class].append((start_time, current_time, pitch, velocity))
                active_notes[instr_class][pitch] = None
            elif token < 16 * 128 * 2 + 32 * 16:
                instr_class = (token-16*128*2) // 32
                velocity = (token-16*128*2) % 32
                current_velocity[instr_class] = velocity
            else:
                current_time += (token-16*128*2-32*16)

        midi = pretty_midi.PrettyMIDI()
        for instr_class, instr_notes in enumerate(unsorted_notes):
            instr_notes.sort()
            if instr_notes != []:
                instr = pretty_midi.Instrument(program=(instr_class*8))
                for (start_time, end_time, pitch, velocity) in instr_notes:
                    note = pretty_midi.Note(
                        start=0.01*start_time,
                        end=0.01*end_time,
                        pitch=pitch,
                        velocity=velocity*4
                    )
                    instr.notes.append(note)
                midi.instruments.append(instr)
        return midi


class LyricsLoader(Loader):
    """Objects of this class parse lyrics files and persist word IDs.

    Arguments:
        max_len (int): maximum length of sequence of words
        metadata (Metadata): a Metadata object
        tokenizer (callable): a callable which takes a file name and returns a
            list of words. Defaults to nltk's `word_tokenize`, which requires
            the punkt tokenizer models. You can download the models with
            `nltk.download('punkt')`
        persist (bool): if true, the tokenizer will persist the IDs of each word
            to a file. If the file already exists, the tokenizer will bootstrap
            from the file.
    """
    def __init__(self, max_len, metadata, tokenizer=nltk.word_tokenize,
            persist=True, dtype=np.int32):
        super(LyricsLoader, self).__init__(max_len, dtype=dtype)
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.word_to_id = {}
        self.id_to_word = {}
        self.highest_word_id = -1
        # read persisted word ids
        if persist:
            log.info('Loading lyrics metadata...')
            for line in self.metadata.lines('word_ids.csv'):
                row = line.rstrip('\n').split(',', 1)
                word_id = int(row[0])
                self.word_to_id[row[1]] = word_id
                self.id_to_word[word_id] = row[1]
                if word_id > self.highest_word_id:
                    self.highest_word_id = word_id

    def is_song(self, filepath):
        return filepath.endswith('.txt')

    def read(self, filepath):
        """Read a file.

        Arguments:
            filepath (str): path to the lyrics file. e.g.
                "/home/user/lyrics_data/tool/lateralus.txt"
        """
        return ''.join(open(filepath, 'r', errors='ignore').readlines())

    def tokenize(self, raw_lyrics):
        """Turns a string of lyrics data into a numpy array of int "word" IDs.

        Arguments:
            raw_lyrics (str): Stringified lyrics data
        """
        tokens = []
        for token in self.tokenizer(raw_lyrics):
            if token not in self.word_to_id:
                self.highest_word_id += 1
                self.word_to_id[token] = self.highest_word_id
                self.id_to_word[self.highest_word_id] = token
                if self.persist:
                    self.metadata.write(
                        'word_ids.csv',
                        '%s,%s\n' % (self.highest_word_id, token)
                    )
            tokens.append(self.word_to_id[token])
        return tokens

    def detokenize(self, numpy_data):
        ret = ''
        for token in numpy_data:
            word = self.id_to_word[token]
            if word == "n't":
                ret += word
            elif word not in string.punctuation and not word.startswith("'"):
                ret += " " + word
            else:
                ret += word
        return "".join(ret).strip()


def resolve_pitch_clashes(sorted_notes):
    num_program_families = int((MAX_MIDI_PROGRAM - MIN_MIDI_PROGRAM + 1) / \
        PROGRAMS_PER_FAMILY)
    no_clash_notes = []
    active_pitch_notes = {}
    for program_family in range(num_program_families):
        active_pitch_notes[program_family + 1] = []

    for quantized_start, quantized_end, program, midi_note in sorted_notes:
        program_family = (program - MIN_MIDI_PROGRAM) // PROGRAMS_PER_FAMILY + 1
        new_active_pitch_notes = [(pitch, end) for pitch, end
            in active_pitch_notes[program_family] if end > quantized_start]
        active_pitch_notes[program_family] = new_active_pitch_notes
        note_pitch = midi_note.pitch
        max_end = 0
        for pitch, end in active_pitch_notes[program_family]:
            if pitch == note_pitch and end > max_end:
                max_end = end
        if max_end >= quantized_end:
            continue
        quantized_start = max(quantized_start, max_end)
        active_pitch_notes[program_family].append((note_pitch, quantized_end))
        no_clash_notes.append((quantized_start, quantized_end, program, midi_note))

    return no_clash_notes


def get_event_list(midi_notes, num_velocity_bins=32):
    no_drum_notes = [(quantized_start, quantized_end, program, midi_note)
        for program, instrument, is_drum, quantized_start, quantized_end, midi_note
            in midi_notes if not is_drum]
    sorted_no_drum_notes = sorted(no_drum_notes, key=lambda element: element[0:3])
    no_clash_notes = resolve_pitch_clashes(sorted_no_drum_notes)

    note_on_set = []
    note_off_set = []
    for index, element in enumerate(no_clash_notes):
        quantized_start, quantized_end, program, midi_note = element
        note_on_set.append((quantized_start, index, program, False))
        note_off_set.append((quantized_end, index, program, True))
    note_events = sorted(note_on_set + note_off_set)

    velocity_bin_size = int(math.ceil(
        (MAX_MIDI_VELOCITY - MIN_MIDI_VELOCITY + 1) / num_velocity_bins))
    num_program_families = int((MAX_MIDI_PROGRAM - MIN_MIDI_PROGRAM + 1) / \
        PROGRAMS_PER_FAMILY)

    current_step = 0
    current_velocity_bin = {}
    for program_family in range(num_program_families):
        current_velocity_bin[program_family + 1] = 0
    events = []

    for step, index, program, is_off in note_events:
        if step > current_step:
            while step > current_step + MAX_SHIFT_STEPS:
                events.append((TIME_SHIFT, MAX_SHIFT_STEPS, 0))
                current_step += MAX_SHIFT_STEPS
            events.append((TIME_SHIFT, step-current_step, 0))
            current_step = step

        note_velocity = no_clash_notes[index][3].velocity
        note_pitch = no_clash_notes[index][3].pitch
        velocity_bin = (note_velocity - MIN_MIDI_VELOCITY) // velocity_bin_size + 1
        program_family = (program - MIN_MIDI_PROGRAM) // PROGRAMS_PER_FAMILY + 1
        if not is_off and velocity_bin != current_velocity_bin[program_family]:
            current_velocity_bin[program_family] = velocity_bin
            events.append((VELOCITY, velocity_bin, program_family))
        if not is_off:
            events.append((NOTE_ON, note_pitch, program_family))
        if is_off:
            events.append((NOTE_OFF, note_pitch, program_family))

    return events


def quantize_notes(midi_notes, steps_per_second=100):
    new_midi_notes = []

    for program, instrument, is_drum, midi_note in midi_notes:
        quantized_start = int(midi_note.start*steps_per_second + 0.5)
        quantized_end = int(midi_note.end*steps_per_second + 0.5)
        if quantized_start == quantized_end:
            quantized_end = quantized_end + 1
        new_midi_notes.append((program, instrument, is_drum, quantized_start,
            quantized_end, midi_note))

    return new_midi_notes


def apply_sustain_control_changes(midi_notes, midi_control_changes,
                                  sustain_control_number=64):
    events = []
    events.extend([(midi_note.start, _NOTE_ON, instrument, midi_note) for
      _1, instrument, _2, midi_note in midi_notes])
    events.extend([(midi_note.end, _NOTE_OFF, instrument, midi_note) for
      _1, instrument, _2, midi_note in midi_notes])

    for _1, instrument, _2, control_change in midi_control_changes:
        if control_change.number != sustain_control_number:
            continue
        value = control_change.value
        if value >= 64:
            events.append((control_change.time, _SUSTAIN_ON, instrument,
                           control_change))
        if value < 64:
            events.append((control_change.time, _SUSTAIN_OFF, instrument,
                           control_change))

    events.sort(key=itemgetter(0))

    active_notes = collections.defaultdict(list)
    sus_active = collections.defaultdict(lambda: False)

    time = 0
    for time, event_type, instrument, event in events:
        if event_type == _SUSTAIN_ON:
            sus_active[instrument] = True
        elif event_type == _SUSTAIN_OFF:
            sus_active[instrument] = False
            new_active_notes = []
            for note in active_notes[instrument]:
                if note.end < time:
                    note.end = time
                else:
                    new_active_notes.append(note)
            active_notes[instrument] = new_active_notes
        elif event_type == _NOTE_ON:
            if sus_active[instrument]:
                new_active_notes = []
                for note in active_notes[instrument]:
                    if note.pitch == event.pitch:
                        note.end = time
                        if note.start == note.end:
                            try:
                                midi_notes.remove(note)
                            except ValueError:
                                continue
                    else:
                        new_active_notes.append(note)
                active_notes[instrument] = new_active_notes
            active_notes[instrument].append(event)
        elif event_type == _NOTE_OFF:
            if sus_active[instrument]:
                pass
            else:
                if event in active_notes[instrument]:
                    active_notes[instrument].remove(event)

    for instrument in active_notes.values():
        for note in instrument:
            note.end = time

    return midi_notes, midi_control_changes


def get_control_changes(midi):
    midi_control_changes = []
    for num_instrument, midi_instrument in enumerate(midi.instruments):
        for midi_control_change in midi_instrument.control_changes:
            midi_control_changes.append((midi_instrument.program, num_instrument,
                                         midi_instrument.is_drum,
                                         midi_control_change))
    return midi_control_changes


def get_notes(midi):
    midi_notes = []
    for num_instrument, midi_instrument in enumerate(midi.instruments):
        for midi_note in midi_instrument.notes:
            midi_notes.append((midi_instrument.program, num_instrument,
                               midi_instrument.is_drum, midi_note))
    return midi_notes
