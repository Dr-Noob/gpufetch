#!/bin/bash -u
# Checks the difference between supported uarchs
# and uarchs that have their topology available
# in file uarch.cpp

uarchs="$(grep 'CHECK_UARCH' uarch.cpp | cut -d',' -f4-5 | grep 'UARCH_GEN' | tr -d ' ' | sort | uniq)"
topos="$(grep 'CHECK_TOPO' uarch.cpp | cut -d',' -f3,4 | grep 'UARCH_' | tr -d ' ' | sort | uniq)"

echo "$uarchs" > /tmp/uarchs.txt
echo "$topos" > /tmp/topos.txt
meld /tmp/uarchs.txt /tmp/topos.txt
rm -f /tmp/uarchs.txt /tmp/topos.txt
