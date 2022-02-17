ME_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ME_NAME="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
ME="$ME_PATH/$ME_NAME"
DATA_ID=conll2003
EP=5
MODEL_DIR="${DATA_ID}_models/ep$EP-dev-slot_by2taskV2Wdec"

git diff > $ME_PATH/code.diff 

cd libs/JointBERT

mkdir $MODEL_DIR

cp $ME $MODEL_DIR
mv $ME_PATH/code.diff $MODEL_DIR/code.diff

python3 main.py --task ${DATA_ID}_slotby2task \
                  --model_type bertcased-seqlabelby2task \
                  --model_dir $MODEL_DIR \
                  --do_train --do_eval  --weight_decay 0.01  \
                  --num_train_epochs $EP |& tee $MODEL_DIR/train.log