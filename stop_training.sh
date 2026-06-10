#!/usr/bin/env bash
# Script stop_training.sh per fermare i processi di training, test e TORCS

echo "Arresto dei processi di addestramento in corso..."

# Ferma gli script di training
pkill -f "train_bc.sh" || echo "Nessun processo train_bc.sh attivo."
pkill -f "train_rl.sh" || echo "Nessun processo train_rl.sh attivo."

# Ferma Behavioral Cloning
pkill -f "behavioral_cloning.py" || echo "Nessun processo behavioral_cloning.py attivo."

# Ferma il training TD3+BC
pkill -f "td3_bc.py" || echo "Nessun processo td3_bc.py attivo."

# Ferma agenti di inferenza
pkill -f "test_agent.py" || echo "Nessun processo test_agent.py attivo."

# Ferma TORCS e l'ambiente Xvfb
pkill -f "torcs" || echo "Nessun processo torcs attivo."
pkill -f "gym_torcs" || echo "Nessun processo gym_torcs attivo."
pkill -f "xvfb-run" || echo "Nessun processo xvfb-run attivo."
pkill -f "Xvfb" || echo "Nessun processo Xvfb attivo."

echo "Tutti i processi sono stati fermati con successo."
