cd ../

current_date=$(TZ="Asia/Seoul" date +"%m%d-%H%M%S")

python src/main.py \
    --config config/alanine/gc-c7axc5.yaml \
    --date $current_date \
    --seed 0 \
    --device cuda:4
