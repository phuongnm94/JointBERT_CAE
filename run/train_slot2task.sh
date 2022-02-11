ME="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ME_NAME="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
ME="$ME/$ME_NAME"
DATA_ID=snips

cd libs/JointBERT

EP=30
MODEL_DIR="${DATA_ID}_models/ep$EP-dev-slot_by2task"
mkdir $MODEL_DIR
cp $ME $MODEL_DIR

python3 main.py --task ${DATA_ID}_slotby2task \
                  --model_type bertseqlabelby2task \
                  --model_dir $MODEL_DIR \
                  --do_eval \
                  --num_train_epochs $EP |& tee $MODEL_DIR/train.log