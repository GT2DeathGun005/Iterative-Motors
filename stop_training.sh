#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  stop_training.sh — Ferma in sicurezza i processi di addestramento
# ═══════════════════════════════════════════════════════════════════════

echo "🛑 Arresto dei processi di addestramento AIcar in corso..."

# Ferma train_all.sh e train_rl.sh
pkill -f "train_all.sh" || echo "Nessun processo train_all.sh attivo."
pkill -f "train_rl.sh" || echo "Nessun processo train_rl.sh attivo."

# Ferma Behavioral Cloning
pkill -f "behavioral_cloning.py" || echo "Nessun processo behavioral_cloning.py attivo."

# Ferma SAC Reinforcement Learning e TD3
pkill -f "sac_rl.py" || echo "Nessun processo sac_rl.py attivo."
pkill -f "td3_bc.py" || echo "Nessun processo td3_bc.py attivo."

# Ferma agenti di inferenza
pkill -f "test_agent.py" || echo "Nessun processo test_agent.py attivo."

# Ferma TORCS e l'ambiente Xvfb
pkill -f "torcs" || echo "Nessun processo torcs attivo."
pkill -f "gym_torcs" || echo "Nessun processo gym_torcs attivo."
pkill -f "xvfb-run" || echo "Nessun processo xvfb-run attivo."
pkill -f "Xvfb" || echo "Nessun processo Xvfb attivo."

echo "✅ Tutti i processi sono stati fermati con successo."
