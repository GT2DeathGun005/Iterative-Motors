#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  stop_training.sh — Ferma in sicurezza i processi di addestramento
# ═══════════════════════════════════════════════════════════════════════

echo "🛑 Arresto dei processi di addestramento AIcar in corso..."

# Ferma train_all.sh
pkill -f "train_all.sh" || echo "Nessun processo train_all.sh attivo."

# Ferma Behavioral Cloning
pkill -f "behavioral_cloning.py" || echo "Nessun processo behavioral_cloning.py attivo."

# Ferma SAC Reinforcement Learning
pkill -f "sac_rl.py" || echo "Nessun processo sac_rl.py attivo."

# Ferma agenti di inferenza
pkill -f "test_agent.py" || echo "Nessun processo test_agent.py attivo."

echo "✅ Tutti i processi sono stati fermati con successo."
