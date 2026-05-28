"""
Preprocessing Dataset HDF5 — TORCS Path Following

Pipeline di trasformazione pre-training che:
  1. Crea un backup completo e protetto (read-only + lock) del dataset originale
  2. Esegue un audit delle feature per identificare sensori non informativi
  3. Droppa la feature distFromStart (indice 29) riducendo il vettore da 30D a 29D

Motivazione del drop di distFromStart:
  - La correlazione con tutte le azioni è quasi-zero (|corr| < 0.05)
  - La discontinuità nativa del simulatore (salto 3608→0 al traguardo) rende
    qualsiasi trasformazione (linearizzazione) incompatibile con l'inference,
    dove il sensore restituisce valori grezzi → train-test mismatch
  - La feature induce la rete ad apprendere profili di velocità posizione-dipendenti
    (rallenta a fine giro, accelera all'inizio) che sono spurii e dannosi

Uso:
    python preprocess_dataset.py                       # default: train_set/laps
    python preprocess_dataset.py --laps_dir percorso   # directory custom
    python preprocess_dataset.py --dry-run              # analisi senza modifiche
"""

import os
import sys
import glob
import shutil
import hashlib
import stat
import argparse
import numpy as np
import h5py
from datetime import datetime


# ──────────────────────────────────────────────────────────────────────
#  Costanti
# ──────────────────────────────────────────────────────────────────────

PRUNE_FEATURE_INDICES = [29]  # Drop only distFromStart in 30D (yields 29D)
BACKUP_DIR_NAME = "dataset_backup"
LOCK_FILE_NAME = ".LOCKED"
PREPROCESSING_VERSION = "4.0"  # v4: drop distFromStart (30D -> 29D)

# Nomi feature (per il report di audit — vettore originale 30D)
FEATURE_NAMES_30D = [
    "angle",
    "track_s0 (-90°)", "track_s1 (-75°)", "track_s2 (-60°)", "track_s3 (-45°)",
    "track_s4 (-30°)", "track_s5 (-20°)", "track_s6 (-15°)", "track_s7 (-10°)",
    "track_s8 (-5°)", "track_s9 (0°)", "track_s10 (5°)", "track_s11 (10°)",
    "track_s12 (15°)", "track_s13 (20°)", "track_s14 (30°)", "track_s15 (45°)",
    "track_s16 (60°)", "track_s17 (75°)", "track_s18 (90°)",
    "trackPos",
    "speedX", "speedY", "speedZ",
    "wheelSpinVel_FL", "wheelSpinVel_FR", "wheelSpinVel_RL", "wheelSpinVel_RR",
    "rpm",
    "distFromStart",
]


# ──────────────────────────────────────────────────────────────────────
#  Guard Rail: Protezione percorso backup
# ──────────────────────────────────────────────────────────────────────

def _assert_not_backup(path: str):
    """Guard rail: impedisce qualsiasi scrittura accidentale nella directory di backup."""
    abs_path = os.path.abspath(path)
    if BACKUP_DIR_NAME in abs_path.split(os.sep):
        raise PermissionError(
            f"GUARD RAIL VIOLATO: tentativo di scrittura nella directory di backup!\n"
            f"  Path bloccato: {abs_path}\n"
            f"  La directory '{BACKUP_DIR_NAME}' è protetta e immutabile."
        )


# ──────────────────────────────────────────────────────────────────────
#  1. Sistema di Backup
# ──────────────────────────────────────────────────────────────────────

def _sha256(filepath: str) -> str:
    """Calcola l'hash SHA-256 di un file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def create_backup(laps_dir: str, project_root: str) -> str:
    """Copia il dataset originale in una directory protetta con verifica di integrità.

    Se il backup esiste già e il lock file è presente, salta la copia.
    """
    backup_root = os.path.join(project_root, BACKUP_DIR_NAME)
    backup_laps = os.path.join(backup_root, "laps")

    # Se il backup esiste già, verifica che sia intatto e salta
    if os.path.exists(backup_root):
        lock_path = os.path.join(backup_root, LOCK_FILE_NAME)
        if os.path.exists(lock_path):
            n_backup = len(glob.glob(os.path.join(backup_laps, "lap_*.h5")))
            print(f"  ✅ Backup già presente e protetto ({n_backup} file)")
            print(f"     Path: {backup_root}")
            return backup_root
        else:
            raise RuntimeError(
                f"Directory backup '{backup_root}' esiste ma manca il lock file.\n"
                f"Stato inconsistente — rimuovi manualmente la directory e riavvia."
            )

    print(f"  📦 Creazione backup in: {backup_root}")
    os.makedirs(backup_laps, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(laps_dir, "lap_*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"Nessun file lap_*.h5 trovato in {laps_dir}")

    print(f"     Copio {len(h5_files)} file...")
    integrity_errors = []

    for src in h5_files:
        dst = os.path.join(backup_laps, os.path.basename(src))
        shutil.copy2(src, dst)

        # Verifica integrità con SHA-256
        src_hash = _sha256(src)
        dst_hash = _sha256(dst)
        if src_hash != dst_hash:
            integrity_errors.append(os.path.basename(src))

        # Imposta permessi read-only sul file di backup
        os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    if integrity_errors:
        raise RuntimeError(
            f"Verifica integrità fallita per {len(integrity_errors)} file:\n"
            f"  {integrity_errors}\n"
            f"Backup INCOMPLETO — i file originali NON sono stati modificati."
        )

    # Lock file come segnale di protezione
    lock_path = os.path.join(backup_root, LOCK_FILE_NAME)
    with open(lock_path, 'w') as f:
        f.write(f"Backup creato: {datetime.now().isoformat()}\n")
        f.write(f"File protetti: {len(h5_files)}\n")
        f.write("NON MODIFICARE O ELIMINARE QUESTA DIRECTORY.\n")
    os.chmod(lock_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    # Imposta la directory di backup come read-only
    os.chmod(backup_laps, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

    print(f"  ✅ Backup completato: {len(h5_files)} file copiati e verificati (SHA-256)")
    print(f"     Permessi: READ-ONLY")
    return backup_root


# ──────────────────────────────────────────────────────────────────────
#  2. Feature Audit
# ──────────────────────────────────────────────────────────────────────

def audit_features(laps_dir: str) -> dict:
    """Analizza tutte le feature del dataset per identificare potenziali problemi."""
    _assert_not_backup(laps_dir)

    h5_files = sorted(glob.glob(os.path.join(laps_dir, "lap_*.h5")))
    all_states = []
    all_actions = []

    for f in h5_files:
        with h5py.File(f, 'r') as h5f:
            all_states.append(h5f['states'][:])
            all_actions.append(h5f['actions'][:])

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    steer = actions[:, 0]

    n_samples = states.shape[0]
    n_features = states.shape[1]

    # Usa i nomi giusti in base alla dimensionalità
    feature_names = FEATURE_NAMES_30D[:n_features] if n_features <= 30 else [f"feat_{i}" for i in range(n_features)]

    print(f"\n  📊 AUDIT FEATURE — {n_samples} campioni, {n_features} feature, {len(h5_files)} giri")
    print(f"  {'─' * 80}")

    results = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_laps': len(h5_files),
        'issues': [],
    }

    LOW_VARIANCE_THRESHOLD = 1e-6
    LOW_RANGE_THRESHOLD = 0.001
    NEAR_ZERO_PCT_THRESHOLD = 99.0

    print(f"\n  {'Feature':<26s} | {'std':>9s} | {'range':>10s} | {'corr(st)':>9s} | {'Status'}")
    print(f"  {'─' * 80}")

    for i in range(n_features):
        col = states[:, i]
        std_val = np.std(col)
        range_val = np.max(col) - np.min(col)
        corr_steer = np.corrcoef(col, steer)[0, 1] if std_val > 0 else 0.0
        near_zero_pct = 100.0 * np.mean(np.abs(col) < 1e-6)

        status = "OK"
        if std_val < LOW_VARIANCE_THRESHOLD:
            status = "⚠️  LOW VARIANCE"
            results['issues'].append({'index': i, 'name': feature_names[i], 'issue': f"varianza quasi-zero ({std_val:.2e})"})
        elif range_val < LOW_RANGE_THRESHOLD:
            status = "⚠️  LOW RANGE"
            results['issues'].append({'index': i, 'name': feature_names[i], 'issue': f"range ristretto ({range_val:.6f})"})
        elif near_zero_pct > NEAR_ZERO_PCT_THRESHOLD:
            status = "⚠️  NEAR CONSTANT"
            results['issues'].append({'index': i, 'name': feature_names[i], 'issue': f"{near_zero_pct:.1f}% quasi-zero"})

        print(f"  [{i:2d}] {feature_names[i]:<22s} | {std_val:9.6f} | {range_val:10.6f} | {corr_steer:+9.4f} | {status}")

    if results['issues']:
        print(f"\n  ⚠️  Feature potenzialmente problematiche:")
        for issue in results['issues']:
            print(f"     [{issue['index']:2d}] {issue['name']}: {issue['issue']}")
    else:
        print(f"\n  ✅ Tutte le {n_features} feature hanno varianza, range e distribuzione adeguati.")

    print(f"\n  📝 NOTA: 'time' e 'damage' non sono presenti nel vettore di stato.")
    print(f"     Sono filtrati a monte da flatten_state() durante la data collection.")

    return results


# ──────────────────────────────────────────────────────────────────────
#  3. Drop distFromStart + Ripristino da Backup
# ──────────────────────────────────────────────────────────────────────

def restore_from_backup(backup_dir: str, laps_dir: str):
    """Ripristina i file originali dal backup prima di applicare una nuova trasformazione.

    Necessario quando una versione precedente del preprocessing ha modificato i file.
    """
    _assert_not_backup(laps_dir)  # laps_dir NON deve essere il backup

    backup_laps = os.path.join(backup_dir, "laps")
    backup_files = sorted(glob.glob(os.path.join(backup_laps, "lap_*.h5")))

    if not backup_files:
        raise FileNotFoundError(f"Nessun file di backup trovato in {backup_laps}")

    print(f"  🔄 Ripristino {len(backup_files)} file dal backup...")

    for src in backup_files:
        dst = os.path.join(laps_dir, os.path.basename(src))
        # Il backup è read-only, copiamo il contenuto
        shutil.copy2(src, dst)
        # Ripristina permessi di scrittura (il backup è read-only ma le copie devono essere scrivibili)
        os.chmod(dst, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

    print(f"  ✅ Ripristino completato: {len(backup_files)} file ripristinati.")


def drop_dist_from_start(laps_dir: str, dry_run: bool = False) -> tuple:
    """Rimuove la colonna distFromStart (indice 29) da tutti i file .h5.

    Riduce il vettore di stato da 30D a 29D.

    Returns:
        (processed, skipped, errors)
    """
    _assert_not_backup(laps_dir)

    h5_files = sorted(glob.glob(os.path.join(laps_dir, "lap_*.h5")))
    processed = 0
    skipped = 0
    errors = 0

    for filepath in h5_files:
        filename = os.path.basename(filepath)
        _assert_not_backup(filepath)

        try:
            with h5py.File(filepath, 'r') as h5f:
                # Skip se già processato con v4
                pv = h5f.attrs.get('preprocessing_version', None)
                if pv == PREPROCESSING_VERSION:
                    skipped += 1
                    continue

                states = h5f['states'][:]
                actions = h5f['actions'][:]
                attrs = dict(h5f.attrs)

            # Verifica che il vettore sia ancora 30D (non già troncato)
            if states.shape[1] != 30:
                print(f"  ⚠️  {filename}: dimensionalità inattesa ({states.shape[1]}D), skip")
                skipped += 1
                continue

            # Drop colonna 29 (distFromStart)
            states_29d = np.delete(states, PRUNE_FEATURE_INDICES, axis=1)

            assert states_29d.shape[1] == 29, f"Shape dopo drop: {states_29d.shape}"

            print(f"  ✅ {filename}: 30D → 29D (drop distFromStart) | "
                  f"{states.shape[0]} campioni")

            if dry_run:
                processed += 1
                continue

            # Scrivi file modificato
            _assert_not_backup(filepath)

            with h5py.File(filepath, 'w') as h5f:
                h5f.create_dataset('states', data=states_29d, compression="gzip")
                h5f.create_dataset('actions', data=actions, compression="gzip")
                for k, v in attrs.items():
                    if k != 'preprocessing_version' and k != 'preprocessing_timestamp':
                        h5f.attrs[k] = v
                h5f.attrs['preprocessing_version'] = PREPROCESSING_VERSION
                h5f.attrs['preprocessing_timestamp'] = datetime.now().isoformat()
                h5f.attrs['dropped_features'] = "distFromStart (idx 29)"

            processed += 1

        except Exception as e:
            errors += 1
            print(f"  ❌ {filename}: errore — {e}")

    return processed, skipped, errors


# ──────────────────────────────────────────────────────────────────────
#  4. Pipeline Principale
# ──────────────────────────────────────────────────────────────────────

def process_dataset(laps_dir: str, project_root: str, dry_run: bool = False):
    """Esegue l'intera pipeline di preprocessing."""
    _assert_not_backup(laps_dir)

    print(f"\n{'=' * 70}")
    print(f"  🔧 PREPROCESSING DATASET v4 — TORCS Path Following")
    print(f"  Directory:  {laps_dir}")
    print(f"  Modalità:   {'DRY-RUN (nessuna modifica)' if dry_run else 'ESECUZIONE'}")
    print(f"  Operazione: Drop distFromStart (30D → 29D)")
    print(f"  Timestamp:  {datetime.now().isoformat()}")
    print(f"{'=' * 70}")

    # ── Fase 1: Verifica/Crea Backup ──
    print(f"\n  📦 FASE 1: BACKUP PROTETTO")
    print(f"  {'─' * 60}")
    backup_path = create_backup(laps_dir, project_root)

    # ── Fase 2: Ripristino da Backup (se i file sono stati modificati da v1, v2 o v3) ──
    # Controlla se i file correnti sono stati toccati da una versione precedente
    sample_file = sorted(glob.glob(os.path.join(laps_dir, "lap_*.h5")))[0]
    with h5py.File(sample_file, 'r') as h5f:
        current_dim = h5f['states'].shape[1]
        current_version = h5f.attrs.get('preprocessing_version', None)

    if current_version is not None and current_version != PREPROCESSING_VERSION:
        print(f"\n  🔄 FASE 2: RIPRISTINO DA BACKUP")
        print(f"  {'─' * 60}")
        print(f"  Rilevata versione preprocessing precedente: v{current_version}")
        print(f"  Ripristino dati originali dal backup prima di applicare v{PREPROCESSING_VERSION}...")
        if not dry_run:
            restore_from_backup(backup_path, laps_dir)
    elif current_dim == 29:
        print(f"\n  ℹ️  I file sono già a 29D — verifica versione...")

    # ── Fase 3: Audit Feature (sui dati originali/ripristinati) ──
    print(f"\n  📊 FASE 3: AUDIT FEATURE")
    print(f"  {'─' * 60}")
    audit_results = audit_features(laps_dir)

    # ── Fase 4: Drop distFromStart ──
    print(f"\n  ✂️  FASE 4: DROP distFromStart (30D → 29D)")
    print(f"  {'─' * 60}")
    print(f"  Motivazione: correlazione con azioni ≈ 0, causa train-test mismatch.\n")

    processed, skipped, errors = drop_dist_from_start(laps_dir, dry_run=dry_run)

    # ── Report Finale ──
    print(f"\n  {'─' * 60}")
    print(f"  📋 REPORT FINALE")
    print(f"     File processati:  {processed}")
    print(f"     File skippati:    {skipped} (già preprocessati)")
    print(f"     Errori:           {errors}")
    print(f"     Backup path:      {backup_path}")

    if dry_run:
        print(f"\n  ⚠️  DRY-RUN: nessun file è stato modificato.")
        print(f"     Esegui senza --dry-run per applicare le trasformazioni.")

    if errors > 0:
        print(f"\n  ⚠️  Attenzione: {errors} file hanno fallito.")

    print(f"\n{'=' * 70}")
    print(f"  Preprocessing v{PREPROCESSING_VERSION} completato.")
    print(f"{'=' * 70}\n")

    return processed, skipped, errors


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing Dataset HDF5 per TORCS Path Following"
    )
    parser.add_argument(
        "--laps_dir", type=str, default="train_set/laps",
        help="Directory contenente i file lap_*.h5 (default: train_set/laps)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Esegui l'analisi senza modificare i file"
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    laps_dir = os.path.abspath(args.laps_dir)

    process_dataset(laps_dir, project_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
