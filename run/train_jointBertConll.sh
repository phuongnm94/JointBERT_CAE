ME_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ME_NAME="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
ME="$ME_PATH/$ME_NAME"
DATA_ID=conll2003
EP=4
git diff > $ME_PATH/code.diff

cd libs/JointBERT
MODEL_DIR="${DATA_ID}_models/ep$EP-optimz"
mkdir $MODEL_DIR
cp $ME $MODEL_DIR
mv $ME_PATH/code.diff $MODEL_DIR

python3 main.py --task ${DATA_ID} \
                  --model_type bertcased \
                  --model_dir $MODEL_DIR \
                  --learning_rate 5e-5    \
                  --do_train --do_eval \
                  --num_train_epochs $EP |& tee $MODEL_DIR/train.log