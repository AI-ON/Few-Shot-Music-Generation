#!/usr/bin/python3
"""A module for lyrics and MIDI dataset loaders."""
import os
import numpy as np
import logging
import nltk

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
    token_count = 0
    for line in open(filepath, 'r', errors='ignore'):
        for token in nltk.word_tokenize(line):
            yield token
            token_count += 1
            if token_count >= max_len:
                return


def tokenize_midi_file(filepath):
    """Turns a MIDI file into a list of event IDs.

    Arguments:
        filepath (str): path to the lyrics file. e.g.
            "/home/user/freemidi_data/Tool/lateralus.mid"
    """
    raise NotImplementedError


class LyricsLoader(object):
    """A class which loads and parses lyrics files into a list of word IDs.

    Arguments:
        length (int): maximum length of tokens, after witch to truncate. Songs
            shorter than `length` are zero padded.
        tokenize (function): a function which takes a single string argument
            and returns a list of integer word IDs.
        persist_file_name: (None or str): if not None, causes LyricsLoader to
            load word IDs from and persist word IDs in the specified file.
    """
    def __init__(self, length, tokenize=tokenize_lyrics_file,
            persist_file_name=None, dtype=np.int32):
        self.length = length
        self.tokenize = tokenize
        self.word_ids = {}
        self.highest_word_id = -1
        self.dtype = dtype

        # read persisted word ids
        if persist_file_name is not None:
            for line in open(persist_file_name, 'r'):
                row = line.rstrip('\n').split(',', 1)
                word_id = int(row[0])
                self.word_ids[row[1]] = word_id
                if word_id > self.highest_word_id:
                    self.highest_word_id = word_id

        if persist_file_name is not None:
            self.persist_file = open(persist_file_name, 'w')
        else:
            self.persist_file = None

    def __call__(self, filepath):
        """This method takes some lyrics data and returns a list of integers
        word IDs for that lyrics data.

        Arguments:
            filepath (str): specifies the path to the file to load.
        """
        tokens = self.tokenize(filepath, self.length)
        word_ids = np.zeros(self.length, dtype=self.dtype)
        for token_index, token in enumerate(tokens):
            if token not in self.word_ids:
                self.highest_word_id += 1
                self.word_ids[token] = self.highest_word_id
                if self.persist_file is not None:
                    self.persist_file.write(
                        '%s,%s\n' % (self.highest_word_id, token))
            word_ids[token_index] = self.word_ids[token]
        return word_ids

    def close(self):
        self.persist_file.close()


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
