import argparse
import os
from sac_rl import ReplayBuffer

def main():
    parser = argparse.ArgumentParser(description="Inietta dataset umano nel Replay Buffer del SAC (Offline-to-Online RL)")
    parser.add_argument("--dataset", type=str, default="train_set/laps", help="Path al dataset HDF5 (file singolo o directory)")
    parser.add_argument("--buffer", type=str, default="train_set/checkpoints/replay_buffer.npz", help="Path al buffer .npz da aggiornare")
    parser.add_argument("--capacity", type=int, default=100000, help="Capacità massima del Replay Buffer")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  💉 EXPERT BUFFER INJECTION UTILITY")
    print(f"{'=' * 60}\n")

    # Inizializza il buffer usando la classe di sac_rl.py
    memory = ReplayBuffer(args.capacity)
    
    # Carica i dati precedenti se esistono (in modo da NON sovrascrivere l'esperienza RL accumulata)
    if os.path.exists(args.buffer):
        print(f"  📂 Caricamento esperienza RL pregressa da: {args.buffer}")
        memory.load(args.buffer)
        print(f"  📊 Dimensione del buffer prima dell'iniezione: {len(memory.buffer)}")
    else:
        print(f"  ⚠️ Nessun buffer esistente in {args.buffer}. Verrà creato da zero.")

    # Inietta i nuovi dati esperti usando il metodo che abbiamo aggiunto prima
    print(f"\n  📥 Avvio caricamento offline-to-online da: {args.dataset}")
    memory.load_expert_data(args.dataset)

    # Salva il buffer su disco
    os.makedirs(os.path.dirname(args.buffer), exist_ok=True)
    memory.save(args.buffer)
    
    print(f"\n  ✅ Dimensione finale del buffer: {len(memory.buffer)}")
    print(f"  💾 Buffer salvato e compresso in: {args.buffer}")
    print("\n  🚀 Puoi ora avviare './train_rl.sh' (senza flag --clean) per riprendere l'addestramento.")
    print("     Il Critic userà istantaneamente le tue dimostrazioni per ottimizzare la policy!\n")

if __name__ == "__main__":
    main()
