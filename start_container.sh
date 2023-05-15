dig +short myip.opendns.com @resolver1.opendns.com > ip.txt
singularity exec --nv -B /fast_scratch_1 -B /fast_scratch_2 -B /data /fast_scratch_1/triumfmlutils/containers/container_base_ml_tf_v0.1.38.sif  /bin/bash
