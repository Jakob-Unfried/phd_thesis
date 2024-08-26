#!/bin/bash
DIR="$(dirname ${BASH_SOURCE[0]})"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTFOLDER="$DIR/out/latest"
mkdir -p $OUTFOLDER

if [[ -f $OUTFOLDER/log ]];
then
    echo "Outfolder contains old data. clean up first!"
    exit 1
fi

MAXTIME=300
N_SITES=20
COMMON_ARGS="-n $N_SITES -l 2 --bestof 5 -f $OUTFOLDER -t $MAXTIME"

python $DIR/benchmark.py -m compose_tenpy -s "SU(2)" -b "fusion_tree" $COMMON_ARGS
python $DIR/benchmark.py -m compose_tenpy -s "U(1)" -b "fusion_tree" $COMMON_ARGS
python $DIR/benchmark.py -m compose_tenpy -s "None" -b "fusion_tree" $COMMON_ARGS
python $DIR/benchmark.py -m compose_tenpy -s "U(1)" -b "abelian" $COMMON_ARGS
python $DIR/benchmark.py -m compose_tenpy -s "None" -b "abelian" $COMMON_ARGS
python $DIR/benchmark.py -m compose_tenpy -s "None" -b "no_symmetry" $COMMON_ARGS

python $DIR/benchmark.py -m compose_numpy -s "None" -b "no_symmetry" $COMMON_ARGS


N_SITES=30
COMMON_ARGS="-n $N_SITES -l 1 --bestof 5 -f $OUTFOLDER -t $MAXTIME"

python $DIR/benchmark.py -m svd_tenpy -s "SU(2)" -b "fusion_tree" $COMMON_ARGS
python $DIR/benchmark.py -m svd_tenpy -s "U(1)" -b "fusion_tree" $COMMON_ARGS
python $DIR/benchmark.py -m svd_tenpy -s "None" -b "fusion_tree" $COMMON_ARGS
python $DIR/benchmark.py -m svd_tenpy -s "U(1)" -b "abelian" $COMMON_ARGS
python $DIR/benchmark.py -m svd_tenpy -s "None" -b "abelian" $COMMON_ARGS
python $DIR/benchmark.py -m svd_tenpy -s "None" -b "no_symmetry" $COMMON_ARGS

python $DIR/benchmark.py -m svd_numpy -s "None" -b "no_symmetry" $COMMON_ARGS
