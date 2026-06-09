#!/bin/bash
# Il seguente script è serve per automatizzare l'avvio di TORCS, 
# in modo da evitare di dover premere manualmente i tasti per avviare la simulazione.
# Lo script utilizza il comando `xte` per inviare sequenze di tasti alla finestra di TORCS,
# simulando le azioni necessarie per avviare una nuova gara in modalità pratica.

# Autostart TORCS: Race -> Practice -> New Race -> Start

xte 'key Return'
xte 'usleep 200000'
xte 'key Return'
xte 'usleep 200000'
xte 'key Down'
xte 'usleep 200000'
xte 'key Down'
xte 'usleep 200000'
xte 'key Down'
xte 'usleep 200000'
xte 'key Down'
xte 'usleep 200000'
xte 'key Down'
xte 'usleep 200000'
xte 'key Return'
xte 'usleep 200000'
xte 'key Return'
