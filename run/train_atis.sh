ME="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ME_NAME="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
ME="$ME/$ME_NAME"

cd libs/JointBERT

EP=10
MODEL_DIR="atis_models/ep$EP"
mkdir $MODEL_DIR
cp $ME $MODEL_DIR

python3 main.py --task atis \
                  --model_type bert \
                  --model_dir $MODEL_DIR \
                  --do_train --do_eval \
                  --num_train_epochs $EP |& tee $MODEL_DIR/train.log