cd ../

current_date=$(TZ="Asia/Seoul" date +"%m%d-%H%M%S")

device="3"

CUDA_VISIBLE_DEVICES=$device python src/main.py \
  --config config/alanine/c7axc5.yaml \
  --date $current_date \
  --seed 0
  # --device cuda:$device
