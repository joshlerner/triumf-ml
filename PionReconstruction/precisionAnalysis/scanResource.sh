#!/bin/bash
# A shell script written to scan latency for GarNet HLS configurations

usage="Usage: $(basename "$0") [-h] [-c] [-b] [-s n] [-e n] [-w n] name resource gpu --- script to scan resources
where: 
    -h show this help text
    -c compile new models
    -b build new models
    -s set the starting number of bits (default: 4)
    -e set the ending number of bits (default: 21)
    -w set the initial precision bit width (default: 16)"

START=4
END=21
WIDTH=16
COMP=false
BUILD=false

while getopts ':hcbs:e:w:' option; do
    case "$option" in
        h) echo "$usage"
            exit
            ;;
        c) COMP=true
            ;;
        b) BUILD=true
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

[ "$#" -eq 3 ] || { printf "2 arguments required, $# provided\n" >&2; echo "$usage" >&2; exit 1; }

NAME="$1Resource"
GPU=$3

filename=${NAME/Fraction}
filename=${filename/Integer}

PRECISIONS=()
while read -r; do
    PRECISIONS+=("$REPLY")
done < <(python3 "PionReconstruction/precisionAnalysis/setupPrecisions.py" "--start" $START "--end" $END "--width" $WIDTH "$1$2")

if $COMP ; then
    rm -r /fast_scratch_1/jlerner/tmp/"$NAME"
    for precision in "${PRECISIONS[@]}"; do
        echo "Compiling $precision"
        for vmax in 32 64; do
            mkdir -p /fast_scratch_1/jlerner/tmp/"$NAME"/"$( echo ${precision//[!0-9]/})"/continuous/"$vmax"
            mkdir -p /fast_scratch_1/jlerner/tmp/"$NAME"/"$( echo ${precision//[!0-9]/})"/quantized/"$vmax"
            echo "$vmax Cells"
            python3 -W ignore "PionReconstruction/precisionAnalysis/${filename}Scan.py" "$NAME" "$vmax" "$precision" $GPU >&2
        done
    done
fi

if $BUILD ; then
    for precision in "${PRECISIONS[@]}"; do
        for vmax in 32 64; do
            echo "Building $precision -- $vmax Cells"
            python3 -W ignore "PionReconstruction/BuildHLS.py" "-s" "--reset" "/fast_scratch_1/jlerner/tmp/$NAME/$( echo ${precision//[!0-9]/})/continuous/$vmax" >&2 &
            python3 -W ignore "PionReconstruction/BuildHLS.py" "-s" "--reset" "/fast_scratch_1/jlerner/tmp/$NAME/$( echo ${precision//[!0-9]/})/quantized/$vmax" >&2 &
        done
    done
    wait
fi

for precision in "${PRECISIONS[@]}"; do
    for vmax in 32 64; do
        python3 "PionReconstruction/precisionAnalysis/parseBuildReport.py" "$NAME" "$1$2" $vmax "$precision"
    done
done

python3 "PionReconstruction/precisionAnalysis/plotPrecisionScan.py" "$1$2"
