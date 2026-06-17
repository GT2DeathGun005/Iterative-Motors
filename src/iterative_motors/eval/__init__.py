"""Valutazione deterministica dell'agente Iterative Motors.

Contiene l'entrypoint ``test_agent`` che esegue la policy in modalità deterministica
(senza rumore esplorativo) su TORCS, con auto-detect del miglior checkpoint disponibile,
e produce la telemetria CSV usata per validare i candidati alla submission.

L'entrypoint si lancia come modulo (``python -m iterative_motors.eval.test_agent``) o tramite
lo script orchestratore ``run.sh``; per questo non viene ri-esportato qui (evita import pesanti
e l'avvio accidentale di dipendenze grafiche all'import del package).
"""
