"""Salvataggio/caricamento checkpoint atomico e resistente alle interruzioni.

Protocollo: scrittura su file temporaneo -> fsync -> rotazione backup (.bak/.prev)
-> os.replace atomico -> fsync della directory. I backup vengono raccolti in
``train_set/checkpoints/backups/`` per tenere pulito l'albero dei file. Logica
spostata verbatim dai monoliti per preservare l'"archivio intoccabile".
"""

import os
import shutil

import torch

from .constants import CHECKPOINT_ROOT, CHECKPOINT_BACKUP_ROOT


def _fsync_file(path):
    """Forza la scrittura fisica su disco del file (evita file a 0 byte su crash)."""
    with open(path, 'rb') as f:
        os.fsync(f.fileno())


def _fsync_dir(path):
    """Sincronizza i metadati della directory (persistenza di os.replace)."""
    dir_fd = os.open(path or '.', os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _backup_paths(filepath):
    """Percorsi (.bak, .prev) per il backup, in ``checkpoints/backups/`` se applicabile."""
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
    """Ruota i backup: .bak -> .prev e copia il file corrente in .bak."""
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
    """Lista ordinata di candidati (primario + backup) per il caricamento robusto."""
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
    """Salva un oggetto PyTorch in modo atomico e resistente alle interruzioni di corrente.

    Protocollo:
      1. salva l'oggetto su un percorso temporaneo (estensione ``.tmp``);
      2. esegue ``fsync`` per forzarne la persistenza fisica sul disco;
      3. ruota i backup esistenti (``.bak`` -> ``.prev``, copia corrente -> ``.bak``);
      4. rinomina atomicamente il temporaneo nel percorso finale con ``os.replace``;
      5. sincronizza i metadati della directory.

    In caso di crash a metà operazione il file finale resta intatto (quello vecchio) o, al più,
    recuperabile dai backup: non si ottiene mai un checkpoint corrotto a 0 byte.
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
    """Scrive testo (sidecar di record) con lo stesso protocollo atomico."""
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
    """Legge un float da un sidecar testuale, con fallback sui backup (.bak/.prev)."""
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
    """Salva un ReplayBuffer (.npz) in modo atomico (usa estensione .tmp.npz)."""
    if len(buffer_obj.buffer) == 0:
        return
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    # np.savez_compressed appende '.npz' se assente: il temp termina con '.tmp.npz'.
    temp_filepath = filepath.replace(".npz", "") + ".tmp.npz"
    buffer_obj.save(temp_filepath)
    if os.path.exists(temp_filepath):
        _fsync_file(temp_filepath)
        if keep_backup:
            _rotate_backup(filepath)
        os.replace(temp_filepath, filepath)
        _fsync_dir(os.path.dirname(filepath) or '.')
