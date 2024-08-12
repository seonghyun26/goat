current_date=$(date +"%m%d-%H%M%S")
seed=0

cd ../

echo Training from c7ax to c5

CUDA_VISIBLE_DEVICES=$1 python src/train.py \
  --start_state c7ax \
  --end_state c5 \
  --date $current_date \
  --seed $seed \
  --wandb

sleep 2

echo Evaluating from c7ax to c5

CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
  --start_state c7ax \
  --end_state c5 \
  --date $current_date \
  --seed $seed \
  --wandb 