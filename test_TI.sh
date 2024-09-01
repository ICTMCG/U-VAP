export BASE_MODEL_PATH="./pretrained_models/<name of pretrained SD>"  # the path of final trained TI checkpoint
export MODEL_PATH="./pretrained_models/<name of pretrained TI checkpoint>"  # the path of final trained TI checkpoint
export NONTAR_MODEL_PATH="./pretrained_models/<name of pretrained nontarget TI checkpoint>"  # the path of final trained nontarget TI checkpoint
export DEVICE=$1
export neg_weight=$2  # degree of adjustment
export omega_a=$3  # guidance scale w_a
export omega=$4   # guidance scale w
export seed=$5  # seed

# 3. inference
pos_token="<colorful-stripe>"
neg_token="<statue>"
prompt="a photo of sks stripe dress"
prompt_ori="a photo of dress"

CUDA_VISIBLE_DEVICES=$DEVICE python scripts/inference_TI_v2.py \
    --prompt "${prompt}" \
    --prompt_ori "${prompt_ori}" \
    --placeholder_token "sks"\
    --pos_token "${pos_token}"\
    --neg_token "${neg_token}"\
    --base_model "${BASE_MODEL_PATH}"\
    --model_path "${MODEL_PATH}"\
    --nontar_model_path "${NONTAR_MODEL_PATH}"\
    --weight "${neg_weight}" \
    --omega_a "${cfg}" \
    --omega "${cfg_new}" \
    --seed "${seed}" \
