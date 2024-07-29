cd ../

current_date=$(TZ="Asia/Seoul" date +"%m%d-%H%M%S")

device="5"

CUDA_VISIBLE_DEVICES=$device python src/main.py \
    --config config/alanine/gc-c7axc5.yaml \
    --date $current_date \
    --seed 0 
