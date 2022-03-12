ME_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ME_NAME="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
ME="$ME_PATH/$ME_NAME"
CUDA_VISIBLE_DEVICES=0
DATA_ID=conll2003truecase
EP=10
git diff > $ME_PATH/code.diff 

cd libs/JointBERT
MODEL_DIR="${DATA_ID}_models/ep$EP-CAE-5e5B32CRF"
mkdir $MODEL_DIR
cp $ME $MODEL_DIR
mv $ME_PATH/code.diff $MODEL_DIR/code.diff

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python3 main.py --task ${DATA_ID}_slotby2task \
                  --model_type bertLcased-seqlabelby2task_ner_ \
                  --model_dir $MODEL_DIR \
                  --max_seq_len 128 --learning_rate 5e-5 --save_steps 100 --logging_steps 100 --train_batch_size 32 --use_crf \
                  --do_train --do_eval  \
                  --num_train_epochs $EP |& tee $MODEL_DIR/train.log

python3 err_analyze.py --model_dir $MODEL_DIR --mode test 
python3 err_analyze.py --model_dir $MODEL_DIR --mode dev 
