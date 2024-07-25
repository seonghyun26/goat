cd ../

current_date=$(date +"%m%d-%H%M%S")

for seed in 0; do
  CUDA_VISIBLE_DEVICES=7 python src/train.py \
    --date $current_date \
    --seed $seed \
    --save_freq 10 \
    --wandb \
    --project goat
done
