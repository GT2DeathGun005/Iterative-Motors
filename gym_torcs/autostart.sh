#!/bin/bash
# Autostart TORCS: Race → Practice → New Race → Start
# Richiede xte (pacchetto xautomation)
# Invia Return, 5 volte Down, e poi 3 Return per avviare Practice Mode.

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











