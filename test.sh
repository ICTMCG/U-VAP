export MODEL_NAME="./pretrained_models/<name of pretrained SD>"  # the path of final trained model
export DEVICE=$1
export neg_weight=$2  # degree of adjustment
export seed=$3  # seed

# 3. inference
pos_token="pos"
neg_token="neg"
prompt="a photo of shirt in sks color"

CUDA_VISIBLE_DEVICES=$DEVICE python scripts/test_db_new_version.py \
    --prompt "${prompt}" \
    --placeholder_token "sks"\
    --pos_token "${pos_token}"\
    --neg_token "${neg_token}"\
    --model_path "${MODEL_PATH}"\
    --type "${type}" \
    --weight "${neg_weight}" \
    --load_words \
    --num_image 50 \
    --seed "${seed}"
