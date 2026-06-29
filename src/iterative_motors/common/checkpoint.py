"""Atomic, interruption-resistant checkpoint save/load.

Protocol: write to a temporary file -> fsync -> backup rotation (.bak/.prev)
-> atomic os.replace -> directory fsync. Backups are collected in
``train_set/checkpoints/backups/`` to keep the file tree clean. Logic moved
verbatim from the monoliths to preserve the "untouchable archive".
"""

import os
import shutil

import torch

from .constants import CHECKPOINT_ROOT, CHECKPOINT_BACKUP_ROOT


def _fsync_file(path):
    """Forces the physical write of the file to disk (avoids 0-byte files on crash)."""
    with open(path, 'rb') as f:
        os.fsync(f.fileno())


def _fsync_dir(path):
    """Syncs the directory metadata (persistence of os.replace)."""
    dir_fd = os.open(path or '.', os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _backup_paths(filepath):
    """Paths (.bak, .prev) for the backup, under ``checkpoints/backups/`` if applicable."""
    abs_filepath = os.path.abspath(filepath)
    backup_base = None
    try:
        if os.path.commonpath([abs_filepath, CHECKPOINT_ROOT]) == CHECKPOINT_ROOT:
            relative_path = os.path.relpath(abs_filepath, CHECKPOINT_ROOT)
            if relative_path != 'backups' and not relative_path.startswith('backups' + os.sep):
                backup_base = os.path.join(CHECKPOINT_BACKUP_ROOT, relative_path)
    except ValueError:
        backup_base = None

    if backup_base is None:
        backup_base = filepath

    return backup_base + ".bak", backup_base + ".prev"


def _rotate_backup(filepath):
    """Rotates the backups: .bak -> .prev and copies the current file to .bak."""
    if not os.path.exists(filepath):
        return
    backup_path, previous_path = _backup_paths(filepath)
    backup_directory = os.path.dirname(backup_path) or '.'
    os.makedirs(backup_directory, exist_ok=True)

    if os.path.exists(backup_path):
        os.replace(backup_path, previous_path)

    temp_backup = backup_path + ".tmp"
    shutil.copy2(filepath, temp_backup)
    _fsync_file(temp_backup)
    os.replace(temp_backup, backup_path)
    _fsync_dir(backup_directory)


def _checkpoint_candidates(filepath):
    """Ordered list of candidates (primary + backups) for robust loading."""
    backup_path, previous_path = _backup_paths(filepath)
    candidates = [filepath, backup_path, previous_path, filepath + ".bak", filepath + ".prev"]
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = os.path.abspath(candidate)
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)
    return unique_candidates


def safe_save(obj, filepath, keep_backup=True):
    """Saves a PyTorch object atomically and resistant to power interruptions.

    Protocol:
      1. save the object to a temporary path (``.tmp`` extension);
      2. ``fsync`` to force its physical persistence to disk;
      3. rotate the existing backups (``.bak`` -> ``.prev``, current copy -> ``.bak``);
      4. atomically rename the temporary to the final path with ``os.replace``;
      5. sync the directory metadata.

    In case of a mid-operation crash the final file stays intact (the old one) or, at most,
    recoverable from the backups: a corrupted 0-byte checkpoint is never produced.
    """
    directory = os.path.dirname(filepath) or '.'
    os.makedirs(directory, exist_ok=True)
    temp_filepath = filepath + ".tmp"
    torch.save(obj, temp_filepath)
    _fsync_file(temp_filepath)
    if keep_backup:
        _rotate_backup(filepath)
    os.replace(temp_filepath, filepath)
    _fsync_dir(directory)


def safe_write_text(filepath, text, keep_backup=True):
    """Writes text (record sidecar) with the same atomic protocol."""
    directory = os.path.dirname(filepath) or '.'
    os.makedirs(directory, exist_ok=True)
    temp_filepath = filepath + ".tmp"
    with open(temp_filepath, 'w', encoding='utf-8') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    if keep_backup:
        _rotate_backup(filepath)
    os.replace(temp_filepath, filepath)
    _fsync_dir(directory)


def safe_read_float(filepath, default):
    """Reads a float from a text sidecar, with fallback on the backups (.bak/.prev)."""
    for candidate in _checkpoint_candidates(filepath):
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, 'r', encoding='utf-8') as f:
                return float(f.read().strip())
        except Exception as e:
            print(f"Impossibile leggere valore numerico da {candidate}: {e}")
    return default


def safe_save_npz(buffer_obj, filepath, keep_backup=True):
    """Saves a ReplayBuffer (.npz) atomically (uses the .tmp.npz extension)."""
    if len(buffer_obj.buffer) == 0:
        return
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    # np.savez_compressed appends '.npz' if absent: the temp ends with '.tmp.npz'.
    temp_filepath = filepath.replace(".npz", "") + ".tmp.npz"
    buffer_obj.save(temp_filepath)
    if os.path.exists(temp_filepath):
        _fsync_file(temp_filepath)
        if keep_backup:
            _rotate_backup(filepath)
        os.replace(temp_filepath, filepath)
        _fsync_dir(os.path.dirname(filepath) or '.')
