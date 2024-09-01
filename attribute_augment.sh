export MODEL_NAME="./pretrained_models/<name of pretrained SD>"  # download the pretrained SD to this path
export DEVICE=$1
export pre_train_steps=$2  # for initial model, 500 is recommended
export get_description=$3  # whether generate new descriptions

dir="./concepts/<your concept images>"
filename=$(basename "$dir")
init_word=$(echo "$filename" | awk -F'__' '{print $NF}')

OUTPUT_DIR="./pre_outputs/<saving position of your pre-learning model>"

############## Pre-learning stage
# if you decide to use accelerate, change the next row into this row
# accelerate launch scripts/train_dreambooth.py \  
CUDA_VISIBLE_DEVICES=$DEVICE python scripts/train_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$dir \
    --output_dir=$OUTPUT_DIR \
    --instance_prompt="a photo of sks" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --checkpointing_steps=$pre_train_steps \
    --max_train_steps=$pre_train_steps \
    --num_class_images=50 \


############## self augmentation
# Generating descriptions by GPT
init_prompt="a photo of sks object in cnc color"  # initial prompt
attri_p="color"  # target attribute
attri_n="object"  # non-target attribute
CUDA_VISIBLE_DEVICES=$DEVICE python utils/run_augmentation.py \
    --model_path "${OUTPUT_DIR}" \
    --init_prompt  "${init_prompt}"\
    --attri_p "${attri_p}" \
    --attri_n "${attri_n}" \
    --n_p 50 \
    --delete 1 \
    --get_description "${get_description}" \


CUDA_VISIBLE_DEVICES=$DEVICE python utils/image_filter_v2.py \
    --model_path "${MODEL_NAME}" \
    --init_prompt  "${init_prompt}" \
    --ori_path "${dir}" \
    --image_path  "${OUTPUT_DIR}/output" \
    --attri_p "${attri_p}" \
    --attri_n "${attri_n}" \
    --n_k  30 \
