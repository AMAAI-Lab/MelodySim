"""Headless SoundFont rendering through sf2_loader's FluidSynth binding.

Keep PrettyMIDI's second-based timing throughout. There is no musicpy round trip,
audio device, temporary MIDI file, or fixed 44.1 kHz synth hidden in this backend.
"""

from functools import lru_cache
import importlib
import math
import os
from ctypes import c_int, c_void_p

import numpy as np


def _binding():
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        module = importlib.import_module("sf2_loader")
        implementation = importlib.import_module(module.sf2_loader.__module__)
        return implementation.fluidsynth, module
    except (ImportError, OSError) as error:
        raise RuntimeError("Cannot load sf2_loader/FluidSynth. Install requirements-augmentation.txt; Linux/macOS also need libfluidsynth.") from error


@lru_cache(maxsize=4)
def available_presets(soundfont):
    _, module = _binding()
    loader = module.sf2_loader(str(soundfont))
    try:
        if not loader.sfid_list or loader.sfid_list[0] < 0:
            raise ValueError(f"Cannot load soundfont: {soundfont}")
        banks = loader.all_instruments()
        if not banks.get(0):
            raise ValueError("A GM-compatible soundfont with melodic presets in bank 0 is required")
        return tuple(sorted(banks[0])), tuple(sorted(banks.get(128, {})))
    finally:
        loader.synth.delete()


@lru_cache(maxsize=2)
def _float_writer(binding):
    return binding.cfunc("fluid_synth_write_float", c_int,
                         ("synth", c_void_p, 1), ("length", c_int, 1),
                         ("left", c_void_p, 1), ("left_offset", c_int, 1), ("left_stride", c_int, 1),
                         ("right", c_void_p, 1), ("right_offset", c_int, 1), ("right_stride", c_int, 1))

def _float_samples(binding, synth, count):
    # Avoid clipping dense mixes before the recorded peak scaling.
    writer = _float_writer(binding)
    samples = np.empty((count, 2), dtype=np.float32)
    if writer is None or writer(synth.synth, count, samples.ctypes.data, 0, 2, samples.ctypes.data, 1, 2) != 0:
        raise RuntimeError("FluidSynth failed to render floating-point samples")
    return samples.mean(axis=1)


def backend_identity():
    binding, _ = _binding()
    numbers = [c_int(), c_int(), c_int()]
    binding.fluid_version(*numbers)
    return {"name": "sf2_loader.fluidsynth-events-v1", "fluidsynth_version": ".".join(str(n.value) for n in numbers)}


def render_midi(midi, soundfont, sample_rate, gain=0.2, tail_seconds=0.5):
    binding, _ = _binding()
    melodic, drums = available_presets(str(soundfont))
    track_channels = []
    melodic_channel, drum_channel = 0, 9
    for track in midi.instruments:
        if track.is_drum:
            track_channels.append(drum_channel)
            drum_channel += 16
        else:
            while melodic_channel % 16 == 9:
                melodic_channel += 1
            track_channels.append(melodic_channel)
            melodic_channel += 1
    # Keep every drum channel distinct from melodic channels (including >16 tracks).
    if len(set(track_channels)) != len(track_channels):
        raise ValueError("Synth channel assignment collision")
    channels = max(16, math.ceil((max(track_channels, default=0) + 1) / 16) * 16)
    synth = binding.Synth(gain=gain, samplerate=float(sample_rate), channels=channels,
                          kwargs={"synth.reverb.active": 0, "synth.chorus.active": 0, "synth.cpu-cores": 1})
    try:
        sfid = synth.sfload(str(soundfont))
        if sfid < 0:
            raise ValueError(f"Cannot load soundfont: {soundfont}")
        events = []
        for track_index, (track, channel) in enumerate(zip(midi.instruments, track_channels)):
            presets = drums if track.is_drum else melodic
            program = track.program if track.program in presets else (0 if track.is_drum and 0 in presets else None)
            if program is None:
                raise ValueError(f"Soundfont has no preset for track {track_index}, bank {128 if track.is_drum else 0}, program {track.program}")
            if synth.program_select(channel, sfid, 128 if track.is_drum else 0, program) != 0:
                raise ValueError(f"FluidSynth could not select track {track_index}'s preset")
            for note in track.notes:
                events.extend([(round(note.start * sample_rate), 3, channel, "noteon", (note.pitch, note.velocity)),
                               (round(note.end * sample_rate), 0, channel, "noteoff", (note.pitch,))])
            for event in track.control_changes:
                # Bank selection is deliberately fixed to GM bank 0 / percussion 128.
                if event.number not in (0, 32):
                    events.append((round(event.time * sample_rate), 1, channel, "cc", (event.number, event.value)))
            for event in track.pitch_bends:
                events.append((round(event.time * sample_rate), 2, channel, "pitch_bend", (event.pitch,)))
        total = max(1, round((midi.get_end_time() + tail_seconds) * sample_rate))
        waveform = np.zeros(total, dtype=np.float32)
        position = 0
        for timestamp, _, channel, kind, values in sorted(events):
            timestamp = min(total, max(position, timestamp))
            if timestamp > position:
                waveform[position:timestamp] = _float_samples(binding, synth, timestamp - position)
                position = timestamp
            getattr(synth, kind)(channel, *values)
        if total > position:
            waveform[position:] = _float_samples(binding, synth, total - position)
        if not np.isfinite(waveform).all() or float(np.max(np.abs(waveform))) <= 1e-4:
            raise ValueError("Synthesized waveform is empty, silent or nonfinite")
        return waveform
    finally:
        synth.delete()
