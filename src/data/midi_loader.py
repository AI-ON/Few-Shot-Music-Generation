#!/usr/bin/python3
"""A module for the MIDI dataset loader.
"""
import logging
import string
import collections
import math
from operator import itemgetter

import pretty_midi

from data.base_loader import Loader

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

    def get_num_tokens(self):
        """Get total number of possible MIDI tokens.

        These are: 128 on/off notes for each of 16 instruments,
        32 velocity buckets for each of 16 instruments,
        and 100 for different time-shifts.
        """
        return 16 * 128 * 2 + 32 * 16 + 100

    def tokenize(self, midi):
        """Turns a MIDI file into a list of event IDs.

        Arguments:
            filepath (str): path to the lyrics file. e.g.
                "/home/user/freemidi_data/Tool/lateralus.mid"
        """
        tokens = []
        midi_notes = get_notes(midi)
        midi_control_changes = get_control_changes(midi)
        midi_notes = apply_sustain_control_changes(midi_notes, midi_control_changes)
        midi_notes = quantize_notes(midi_notes)
        no_drum_notes = remove_drums(midi_notes)
        no_clash_notes = resolve_pitch_clashes(no_drum_notes)
        events = get_event_list(no_clash_notes)
        for event_type, event_value, family in events:
            if event_type == NOTE_ON:
                token = family * 128 + event_value
            elif event_type == NOTE_OFF:
                token = 16 * 128 + family * 128 + event_value
            elif event_type == VELOCITY:
                token = 16 * 128 * 2 + 32 * family + event_value
            elif event_type == TIME_SHIFT:
                # subtract one, because TIME_SHIFT event values are 1-indexed
                token = 16 * 128 * 2 + 32 * 16 + event_value - 1
            tokens.append(token)
        return tokens

    def detokenize(self, numpy_data):
        current_time = 0
        current_velocity = [16 for _ in range(16)]
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
                if active_notes[instr_class][pitch] is not None:
                    (velocity, start_time) = active_notes[instr_class][pitch]
                    unsorted_notes[instr_class].append((start_time, current_time, pitch, velocity))
                    active_notes[instr_class][pitch] = None
            elif token < 16 * 128 * 2 + 32 * 16:
                instr_class = (token-16*128*2) // 32
                velocity = (token-16*128*2) % 32
                current_velocity[instr_class] = velocity
            else:
                current_time += (token-16*128*2-32*16+1)

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


def resolve_pitch_clashes(midi_notes):
    """This function resolve note conflicts resulting from merging instruments
    of the same class.

    MIDI specifies 16 instrument classes, with 8 instruments per class. For
    this project, we merge together all instruments for a class into a single
    instrument. This can create issues if you have multiple instruments of the
    same class in the same song (e.g. two electric guitars, or a viola and a
    violin). The conflict occurs when you have two instruments play the same
    note at the same time. For example, if you have guitar 1 begin note 55 at
    time-step 100, and then guitar 2 begin note 55 at time-step 101, and then
    guitar 1 end note 55 at time-step 102, it becomes unclear how to represent
    that as a single instrument. Does note 55 end at the guitar 1 note end
    event? Or does it wait until guitar 2 ends? Does it play the overlapping
    notes as a single note, or does it try to split it into two notes somehow?

    This code solves that problem by allowing the first note to finish. If the
    duration of the second note extends beyond the duration of the first, the
    remaining duration will be played after the first note ends.

    Arguments:
        midi_notes ([(int, int, int, pretty_midi.Note)]): a tuple list of
            information on the MIDI notes of all instruments in a song. The
            first element is the quantized start time of the note. The second
            element is the quantized end time of the note. The third element is
            the instrument number (refer to the General MIDI spec for more
            info).
    """
    num_program_families = int((MAX_MIDI_PROGRAM - MIN_MIDI_PROGRAM + 1) / \
        PROGRAMS_PER_FAMILY)
    no_clash_notes = []
    active_pitch_notes = {}

    sorted_notes = sorted(midi_notes, key=lambda element: element[0:3])
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

def remove_drums(midi_notes):
    """Removes all drum notes from a sequence of MIDI notes.

    Argument:
        midi_notes ([(int, int, bool, int, int, pretty_midi.Note)]): a list
            containing tuples of info on each MIDI note. The first and second
            elements are discarded. The third element is a boolean representing
            if the note is a drum or not. The fourth and fifth are the start
            and end time respectively. The last is the note.
    """
    return [(start, end, program, midi_note)
        for program, instrument, is_drum, start, end, midi_note
            in midi_notes if not is_drum]

def get_event_list(midi_notes, num_velocity_bins=32):
    """Transforms a sequence of MIDI notes into a sequence of events.

    Arguments:
        midi_notes ([(int, int, int, pretty_midi.Note)]): A list containing
            info on each MIDI note.
        num_velocity_bins (int): the number of bins to split the velocity
            into. The MIDI standardizes on 128 possible values (0-127) but
            we bucket subranges together to reduce the dimensionality. Must
            evenly divide into 128.
    """
    note_on_set = []
    note_off_set = []
    for index, element in enumerate(midi_notes):
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

        note_velocity = midi_notes[index][3].velocity
        note_pitch = midi_notes[index][3].pitch
        velocity_bin = (note_velocity - MIN_MIDI_VELOCITY) // velocity_bin_size + 1
        program_family = (program - MIN_MIDI_PROGRAM) // PROGRAMS_PER_FAMILY + 1
        if not is_off and velocity_bin != current_velocity_bin[program_family]:
            current_velocity_bin[program_family] = velocity_bin
            # NOTE: velocity is set per-program-family, but that's not strictly
            # necessary. We could have set-velocity events set the velocity
            # for all instruments. This would change the required number of
            # set-velocity events, but it would reduce the dimensionality of
            # the encoding.
            events.append((VELOCITY, velocity_bin, program_family))
        if not is_off:
            events.append((NOTE_ON, note_pitch, program_family))
        if is_off:
            events.append((NOTE_OFF, note_pitch, program_family))

    return events


def quantize_notes(midi_notes, steps_per_second=100):
    """Quantize MIDI notes into integers. The unit represents a unit of time,
    determined by `steps_per_second`.

    midi_notes ([(int, int, bool, pretty_midi.Note)]): A list containing tuples
        of info describing individual MIDI notes in a song. The first element
        is the MIDI instrument number of the note. The second element is an
        identifier of the MIDI instrument, unique to all instruments within
        the song. The third element is a flag indicating whether the instrument
        is a drum or not. The last element is the note object.
    steps_per_second (int): The number of steps per second. Which each note
        gets rounded toward.
    """
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
    """Applies sustain to the MIDI notes by modifying the notes in-place.

    Normally, MIDI note start/end times simply describe e.g. when a piano key
    is pressed. It's possible that the sound from the note continues beyond
    the pressing of the note if a sustain on the instrument is active. The
    activity of sustain on MIDI instruments is determined by certain control
    events. This function alters the start/end time of MIDI notes with respect
    to the sustain control messages to mimic sustain.

    Arguments:
        midi_notes ([(int, int, bool, pretty_midi.Note)]): A list of tuples of
            info on each MIDI note.
        midi_control_changes ([(int, int, bool, pretty_midi.ControlChange)]):
            A list of tuples on each control change event.
    """
    events = []
    events.extend([(midi_note.start, _NOTE_ON, instrument, midi_note) for
      _1, instrument, _2, midi_note in midi_notes])
    events.extend([(midi_note.end, _NOTE_OFF, instrument, midi_note) for
      _1, instrument, _2, midi_note in midi_notes])

    for _1, instrument, _2, control_change in midi_control_changes:
        if control_change.number != sustain_control_number:
            continue
        value = control_change.value
        # MIDI spec specifies that >= 64 means ON and < 64 means OFF.
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

    return midi_notes


def get_control_changes(midi):
    """Retrieves a list of control change events from a given MIDI song.

    Arguments:
        midi (PrettyMIDI): The MIDI song.
    """
    midi_control_changes = []
    for num_instrument, midi_instrument in enumerate(midi.instruments):
        for midi_control_change in midi_instrument.control_changes:
            midi_control_changes.append((
                midi_instrument.program,
                num_instrument,
                midi_instrument.is_drum,
                midi_control_change
            ))
    return midi_control_changes


def get_notes(midi):
    """Retrieves a list of MIDI notes (for all instruments) given a MIDI song.

    Arguments:
        midi (PrettyMIDI): The MIDI song.
    """
    midi_notes = []
    for num_instrument, midi_instrument in enumerate(midi.instruments):
        for midi_note in midi_instrument.notes:
            midi_notes.append((
                midi_instrument.program,
                num_instrument,
                midi_instrument.is_drum,
                midi_note
            ))
    return midi_notes
