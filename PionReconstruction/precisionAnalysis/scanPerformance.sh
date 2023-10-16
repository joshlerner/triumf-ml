#!/bin/bash
# A shell script written to scan performance for GarNet HLS configurations

usage="Usage: $(basename "$0") [-h] [-s n] [-e n] [-w n] name gpu --- script to scan performance
where: 
    -h show this help text
    -s set the starting number of bits (default: 4)
    -e set the ending number of bits (default: 21)
    -w set the initial precision bit width (default: 16)"

START=4
END=21
WIDTH=16

while getopts ':hs:e:w:' option; do
    case "$option" in
        h) echo "$usage"
            exit
            ;;
        s) START=$OPTARG
            ;;
        e) END=$OPTARG
            ;;
        w) WIDTH=$OPTARG
            ;;
        :) printf "missing argument for -%s\n" "$OPTARG" >&2
            echo "$usage" >&2
            exit 1
            ;;
        \?) printf "illegal option: -%s\n" "$OPTARG" >&2
            echo "$usage" >&2
            exit 1
            ;;
    esac
done

shift $((OPTIND - 1))

[ "$#" -eq 2 ] || { printf "2 arguments required, $# provided\n" >&2; echo "$usage" >&2; exit 1; }

NAME="$1Performance"
GPU=$2

PRECISIONS=()
while read -r; do
    PRECISIONS+=("$REPLY")
done < <(python3 "PionReconstruction/precisionAnalysis/setupPrecisions.py" "--start" $START "--end" $END "--width" $WIDTH $NAME)

filename=${NAME/Fraction}
filename=${filename/Integer}

mkdir -p /fast_scratch_1/jlerner/tmp/$NAME

for precision in "${PRECISIONS[@]}"; do
    echo $precision
    for vmax in 32 64 128; do
        python3 -W ignore "PionReconstruction/precisionAnalysis/${filename}Scan.py" $NAME $vmax "$precision" $GPU >&2
    done
done

python3 "PionReconstruction/precisionAnalysis/plotPrecisionScan.py" $NAME

rm -r /fast_scratch_1/jlerner/tmp/$NAME

