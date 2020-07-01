Make#!/bin/bash
export PATH=/Soft/cuda/9.0.176/bin:$PATH
### Directivas para el gestor de colas
# Asegurar que el job se ejecuta en el directorio actual
#$ -cwd
# Asegurar que el job mantiene las variables de entorno del shell lamador
#$ -V
# Cambiar el nombre del job
#$ -N mergesort
# Cambiar el shell
#$ -S /bin/bash

./mergesort.exe
nvprof -V
nvprof --unified-memory-profiling off ./mergesort.exe
nvprof --unified-memory-profiling off --print-gpu-summary ./mergesort.exe

