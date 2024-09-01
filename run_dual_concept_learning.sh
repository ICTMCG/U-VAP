export MODEL_PATH="./pretrained_models/<name of pretrained SD>"  # download the pretrained SD to this path
export DEVICE=$1
export finetune_steps=$2  # 1000 is recommended

FINAL_DIR="./outputs/<your final model path name>"
dir="./augmented_data/<your augmented data path>"  # put target/non-target images together

############## Dual concept learning stage
# if you decide to use accelerate, change the next row into this row
# accelerate launch scripts/train_dreambooth_ag.py \
CUDA_VISIBLE_DEVICES=$DEVICE python scripts/train_dreambooth_ag.py \
    --pretrained_model_name_or_path=$MODEL_PATH \
    --instance_data_dir=$dir \
    --output_dir=$FINAL_DIR \
    --instance_prompt="{}" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=5 \
    --checkpointing_steps=$finetune_steps \
    --max_train_steps=$finetune_steps
