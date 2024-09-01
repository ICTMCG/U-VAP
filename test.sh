export MODEL_PATH="./pretrained_models/<name of pretrained SD>"  # the path of final trained model
export DEVICE=$1
export neg_weight=$2  # degree of adjustment
export omega_a=$3  # guidance scale w_a
export omega=$4   # guidance scale w
export seed=$5  # seed

# 3. inference
pos_token="pos"
neg_token="neg"
prompt="a photo of backpack in sks color"
prompt_ori="a photo of backpack"

CUDA_VISIBLE_DEVICES=$DEVICE python scripts/inference_v2.py \
    --prompt "${prompt}" \
    --prompt_ori "${prompt_ori}" \
    --placeholder_token "sks"\
    --pos_token "${pos_token}"\
    --neg_token "${neg_token}"\
    --model_path "${MODEL_PATH}"\
    --weight "${neg_weight}" \
    --omega_a "${omega_a}" \
    --omega "${omega}" \
    --seed "${seed}" \
