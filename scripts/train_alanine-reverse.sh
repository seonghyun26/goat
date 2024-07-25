cd ../

current_date=$(date +"%m%d-%H%M%S")

for seed in 0; do
  CUDA_VISIBLE_DEVICES=6 python src/train.py \
    --start_state c7ax \
    --end_state c5 \
    --date $current_date \
    --seed $seed \
    --wandb \
    --project goat
done
