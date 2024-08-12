current_date=$(date +"%m%d-%H%M%S")
seed=0

cd ../

echo Training from c5 to c7ax

CUDA_VISIBLE_DEVICES=$1 python src/train.py \
  --start_state c5 \
  --end_state c7ax \
  --date $current_date \
  --seed $seed \
  --wandb

sleep 2

echo Evaluating from c5 to c7ax

CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
  --start_state c5 \
  --end_state c7ax \
  --date $current_date \
  --seed $seed \
  --wandb 