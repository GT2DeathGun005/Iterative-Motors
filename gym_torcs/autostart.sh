#!/bin/bash
# Autostart TORCS: Race -> Practice -> New Race -> Start
# Richiede xte (pacchetto xautomation)
# Sequenza tasti usata dal menu TORCS per entrare in Practice Mode.

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
