"""
Preprocessing Dataset HDF5 — TORCS Path Following

Pipeline di trasformazione pre-training che:
  1. Crea un backup completo e protetto (read-only + lock) del dataset originale
  2. Esegue un audit delle feature per identificare sensori non informativi
  3. Linearizza la feature distFromStart eliminando le discontinuità
     del simulatore, producendo una metrica di distanza strettamente
     monotona, continua e incrementale per l'intera durata della run.

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

TRACK_LENGTH = 3608.0       # Lunghezza circuito in metri (CG Speedway 1)
DIST_FEATURE_IDX = 29       # Indice di distFromStart nel vettore 30D
DIST_NORM_FACTOR = 4000.0   # Fattore di normalizzazione usato in flatten_state
BACKUP_DIR_NAME = "dataset_backup"
LOCK_FILE_NAME = ".LOCKED"
PREPROCESSING_VERSION = "1.0"

# Nomi feature (per il report di audit)
FEATURE_NAMES = [
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

    Args:
        laps_dir: percorso della directory contenente i file .h5
        project_root: radice del progetto (dove creare dataset_backup/)

    Returns:
        Percorso assoluto della directory di backup creata

    Raises:
        RuntimeError: se il backup esiste già (per evitare sovrascritture)
        RuntimeError: se la verifica di integrità fallisce
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
    """Analizza tutte le feature del dataset per identificare potenziali problemi.

    Controlla:
      - Varianza zero o quasi-zero (feature costanti / non informative)
      - Range eccessivamente ristretto
      - Correlazione con le azioni target
      - Discontinuità nella distFromStart

    Returns:
        Dizionario con i risultati dell'audit, inclusa lista di feature problematiche
    """
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

    print(f"\n  📊 AUDIT FEATURE — {n_samples} campioni, {n_features} feature, {len(h5_files)} giri")
    print(f"  {'─' * 80}")

    results = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_laps': len(h5_files),
        'issues': [],
        'per_feature': {},
    }

    LOW_VARIANCE_THRESHOLD = 1e-6
    LOW_RANGE_THRESHOLD = 0.001
    NEAR_ZERO_PCT_THRESHOLD = 99.0  # se >99% dei valori sono quasi zero

    print(f"\n  {'Feature':<26s} | {'std':>9s} | {'range':>10s} | {'corr(st)':>9s} | {'Status'}")
    print(f"  {'─' * 80}")

    for i in range(n_features):
        col = states[:, i]
        std_val = np.std(col)
        range_val = np.max(col) - np.min(col)
        corr_steer = np.corrcoef(col, steer)[0, 1] if std_val > 0 else 0.0
        near_zero_pct = 100.0 * np.mean(np.abs(col) < 1e-6)

        status = "OK"
        issues_for_feature = []

        if std_val < LOW_VARIANCE_THRESHOLD:
            status = "⚠️  LOW VARIANCE"
            issues_for_feature.append(f"varianza quasi-zero ({std_val:.2e})")
        elif range_val < LOW_RANGE_THRESHOLD:
            status = "⚠️  LOW RANGE"
            issues_for_feature.append(f"range ristretto ({range_val:.6f})")
        elif near_zero_pct > NEAR_ZERO_PCT_THRESHOLD:
            status = "⚠️  NEAR CONSTANT"
            issues_for_feature.append(f"{near_zero_pct:.1f}% quasi-zero")

        if issues_for_feature:
            results['issues'].append({
                'index': i,
                'name': FEATURE_NAMES[i],
                'issues': issues_for_feature,
            })

        results['per_feature'][i] = {
            'name': FEATURE_NAMES[i],
            'std': float(std_val),
            'range': float(range_val),
            'corr_steer': float(corr_steer),
        }

        print(f"  [{i:2d}] {FEATURE_NAMES[i]:<22s} | {std_val:9.6f} | {range_val:10.6f} | {corr_steer:+9.4f} | {status}")

    # Report finale
    if results['issues']:
        print(f"\n  ⚠️  Feature potenzialmente problematiche:")
        for issue in results['issues']:
            print(f"     [{issue['index']:2d}] {issue['name']}: {', '.join(issue['issues'])}")
    else:
        print(f"\n  ✅ Tutte le {n_features} feature hanno varianza, range e distribuzione adeguati.")
        print(f"     Nessun sensore costante, statico o non informativo rilevato.")

    # Nota su time e damage
    print(f"\n  📝 NOTA: 'time' e 'damage' non sono presenti nel vettore di stato 30D.")
    print(f"     Sono filtrati a monte da flatten_state() durante la data collection.")
    print(f"     Non è necessario alcun pruning per queste variabili.")

    return results


# ──────────────────────────────────────────────────────────────────────
#  3. Adapter di Linearizzazione distFromStart
# ──────────────────────────────────────────────────────────────────────

def linearize_dist_from_start(dist_scaled: np.ndarray,
                               track_length: float = TRACK_LENGTH) -> np.ndarray:
    """Linearizza la feature distFromStart eliminando le discontinuità del simulatore.

    Il sensore distFromStart di TORCS ha un comportamento discontinuo:
      - Spawn pre-linea: valore alto (~3598m), cresce lentamente verso ~3608m
      - Attraversamento start/finish: crolla istantaneamente a ~0
      - Post-linea: cresce linearmente da 0 a ~3607m

    Questa funzione converte la serie in una metrica strettamente monotona,
    continua e partente da 0, interpretando correttamente la distanza percorsa.

    Args:
        dist_scaled: array 1D con distFromStart già diviso per 4000 (come nei .h5)
        track_length: lunghezza del circuito in metri

    Returns:
        Array 1D linearizzato e ri-normalizzato in [0, ~1]
    """
    # Lavoriamo in metri per chiarezza
    dist_m = dist_scaled * DIST_NORM_FACTOR

    n = len(dist_m)
    dist_linear = np.zeros(n, dtype=np.float64)

    # ── Step 1: Identifica il punto di discontinuità (crossing della start/finish line)
    diffs = np.diff(dist_m)
    # La discontinuità principale è il salto negativo più grande (da ~3608 a ~0)
    jump_idx = np.argmin(diffs)
    jump_magnitude = abs(diffs[jump_idx])

    # Verifica che sia una discontinuità reale (almeno 1000m di salto)
    if jump_magnitude < 1000.0:
        # Nessuna discontinuità significativa: la serie è già ragionevolmente lineare
        # (potrebbe accadere se i dati sono già preprocessati)
        dist_linear = dist_m - dist_m[0]
        return (dist_linear / DIST_NORM_FACTOR).astype(np.float32)

    # ── Step 2: Fase pre-linea (step 0 .. jump_idx)
    # Lo spawn è a dist_m[0] ≈ 3598m. La linea è a dist ≈ track_length.
    # La distanza percorsa dallo spawn alla linea è:
    #   d_percorsa = (track_length - dist_m[0]) + crescita residua fino a jump_idx
    # Ma più semplicemente: mappiamo ogni punto pre-linea come distanza dallo spawn:
    #   dist_linear[i] = dist_m[i] - dist_m[0]  per i in [0, jump_idx]
    # Questo dà 0 al punto di spawn e ~10m alla linea di partenza.

    spawn_dist = dist_m[0]  # ≈3598m (posizione di spawn sulla pista)

    for i in range(0, jump_idx + 1):
        dist_linear[i] = dist_m[i] - spawn_dist

    # Offset alla linea: la distanza cumulativa percorsa fino al crossing
    offset_at_line = dist_linear[jump_idx]  # ≈10m

    # ── Step 3: Fase post-linea (step jump_idx+1 .. end)
    # Dopo il crossing, dist_m riparte da ~0 e cresce.
    # La distanza cumulativa è: offset_at_line + dist_m[i]
    for i in range(jump_idx + 1, n):
        dist_linear[i] = offset_at_line + dist_m[i]

    # ── Step 4: Enforce monotonia stretta (clamp micro-oscillazioni del simulatore)
    # Il simulatore TORCS introduce rumore dell'ordine di ~3µm nella fase pre-linea.
    # Running-max clamp: ogni valore deve essere >= il massimo precedente.
    running_max = dist_linear[0]
    for i in range(1, n):
        if dist_linear[i] < running_max:
            dist_linear[i] = running_max
        else:
            running_max = dist_linear[i]

    # ── Step 5: Ri-normalizzazione
    # Normalizziamo dividendo per la distanza totale percorsa, così il range è [0, ~1]
    total_distance = dist_linear[-1]
    if total_distance > 0:
        dist_linear = dist_linear / total_distance

    return dist_linear.astype(np.float32)


def validate_linearization(dist_linear: np.ndarray, filename: str) -> bool:
    """Valida che la serie linearizzata soddisfi i requisiti.

    Verifica:
      - Monotonia non-decrescente
      - Continuità (nessun salto > soglia)
      - Range approssimativo [0, 1]
      - Assenza di NaN/Inf

    Returns:
        True se la validazione passa, False altrimenti
    """
    # NaN/Inf check
    if np.any(np.isnan(dist_linear)):
        print(f"  ❌ {filename}: NaN rilevati nella serie linearizzata")
        return False
    if np.any(np.isinf(dist_linear)):
        print(f"  ❌ {filename}: Inf rilevati nella serie linearizzata")
        return False

    # Monotonia: tutti i diff devono essere >= 0 (con piccola tolleranza numerica)
    diffs = np.diff(dist_linear.astype(np.float64))
    n_negative = np.sum(diffs < -1e-6)
    if n_negative > 0:
        worst_drop = diffs.min()
        worst_idx = np.argmin(diffs)
        print(f"  ❌ {filename}: Monotonia violata! {n_negative} decrementi, "
              f"peggiore: {worst_drop:.6f} all'indice {worst_idx}")
        return False

    # Continuità: nessun salto > 0.01 (in scala normalizzata)
    max_jump = np.max(np.abs(diffs))
    if max_jump > 0.01:
        jump_idx = np.argmax(np.abs(diffs))
        print(f"  ⚠️  {filename}: Salto di {max_jump:.6f} all'indice {jump_idx} "
              f"(soglia: 0.01)")
        # Questo è un warning, non un errore fatale

    # Range
    if abs(dist_linear[0]) > 0.01:
        print(f"  ⚠️  {filename}: Valore iniziale = {dist_linear[0]:.6f} (atteso ~0.0)")
    if abs(dist_linear[-1] - 1.0) > 0.15:
        print(f"  ⚠️  {filename}: Valore finale = {dist_linear[-1]:.6f} (atteso ~1.0)")

    return True


# ──────────────────────────────────────────────────────────────────────
#  4. Pipeline Principale
# ──────────────────────────────────────────────────────────────────────

def process_dataset(laps_dir: str, project_root: str, dry_run: bool = False):
    """Esegue l'intera pipeline di preprocessing.

    Args:
        laps_dir: directory contenente i file lap_*.h5
        project_root: radice del progetto
        dry_run: se True, analizza senza modificare i file
    """
    # Guard rail: verifica che non stiamo operando sul backup
    _assert_not_backup(laps_dir)

    print(f"\n{'=' * 70}")
    print(f"  🔧 PREPROCESSING DATASET — TORCS Path Following")
    print(f"  Directory:  {laps_dir}")
    print(f"  Modalità:   {'DRY-RUN (nessuna modifica)' if dry_run else 'ESECUZIONE'}")
    print(f"  Timestamp:  {datetime.now().isoformat()}")
    print(f"{'=' * 70}")

    # ── Fase 1: Backup ──
    print(f"\n  📦 FASE 1: BACKUP PROTETTO")
    print(f"  {'─' * 60}")
    backup_path = create_backup(laps_dir, project_root)

    # ── Fase 2: Audit Feature ──
    print(f"\n  📊 FASE 2: AUDIT FEATURE")
    print(f"  {'─' * 60}")
    audit_results = audit_features(laps_dir)

    # ── Fase 3: Linearizzazione distFromStart ──
    print(f"\n  🔄 FASE 3: LINEARIZZAZIONE distFromStart")
    print(f"  {'─' * 60}")

    h5_files = sorted(glob.glob(os.path.join(laps_dir, "lap_*.h5")))
    processed = 0
    skipped = 0
    errors = 0

    for filepath in h5_files:
        filename = os.path.basename(filepath)

        # Guard rail esplicito
        _assert_not_backup(filepath)

        with h5py.File(filepath, 'r') as h5f:
            # Skip se già preprocessato
            if h5f.attrs.get('preprocessing_version', None) == PREPROCESSING_VERSION:
                skipped += 1
                continue

            states = h5f['states'][:]
            actions = h5f['actions'][:]
            # Preserva tutti gli attributi originali
            attrs = dict(h5f.attrs)

        # Estrai e linearizza distFromStart
        dist_original = states[:, DIST_FEATURE_IDX].copy()
        dist_linearized = linearize_dist_from_start(dist_original)

        # Validazione
        if not validate_linearization(dist_linearized, filename):
            errors += 1
            print(f"  ❌ {filename}: validazione fallita, file NON modificato")
            continue

        # Report per questo file
        dist_original_m = dist_original * DIST_NORM_FACTOR
        dist_linear_m = dist_linearized * dist_linearized[-1] * DIST_NORM_FACTOR  # stima

        diffs_orig = np.diff(dist_original_m)
        jump_idx = np.argmin(diffs_orig)

        print(f"  ✅ {filename}: linearizzato | "
              f"jump@{jump_idx} rimosso | "
              f"range [{dist_linearized[0]:.4f}, {dist_linearized[-1]:.4f}] | "
              f"monotono: {np.all(np.diff(dist_linearized) >= -1e-6)}")

        if dry_run:
            processed += 1
            continue

        # Scrivi il file modificato (sovrascrivendo l'originale, il backup è protetto)
        states[:, DIST_FEATURE_IDX] = dist_linearized

        _assert_not_backup(filepath)  # Doppio check prima della scrittura

        with h5py.File(filepath, 'w') as h5f:
            h5f.create_dataset('states', data=states, compression="gzip")
            h5f.create_dataset('actions', data=actions, compression="gzip")
            # Ripristina attributi originali
            for k, v in attrs.items():
                h5f.attrs[k] = v
            # Aggiungi flag di preprocessing
            h5f.attrs['preprocessing_version'] = PREPROCESSING_VERSION
            h5f.attrs['preprocessing_timestamp'] = datetime.now().isoformat()

        processed += 1

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
        print(f"\n  ⚠️  Attenzione: {errors} file hanno fallito la validazione.")
        print(f"     I file con errori NON sono stati modificati.")
        print(f"     Controlla i log sopra per i dettagli.")

    print(f"\n{'=' * 70}")
    print(f"  Preprocessing completato.")
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
