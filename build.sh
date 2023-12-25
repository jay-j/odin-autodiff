#!/bin/bash
set -x

odin build . -out:prog.bin -debug -show-timings -o:speed
