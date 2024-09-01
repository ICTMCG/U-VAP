export MODEL_PATH="./pretrained_models/<name of pretrained SD>"  # download the pretrained SD to this path
export DEVICE=$1
export finetune_steps=$2  # 3000 is recommended

FINAL_DIR="./outputs_TI/<your target final model path name>"
dir="./augmented_data_TI/<your target augmented data path>"  # put target images together
NONTAR_FINAL_DIR="./outputs_TI/<your nontarget final model path name>"
nontar_dir="./augmented_data_TI/<your nontarget augmented data path>"  # put non-target images together

CUDA_VISIBLE_DEVICES=$DEVICE accelerate launch --main_process_port $port scripts/textual_inversion.py \
    --pretrained_model_name_or_path=$MODEL_PATH \
    --train_data_dir=$dir \
    --learnable_property="object" \
    --placeholder_token="<colorful-stripe>" \
    --initializer_token="stripe" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=$finetune_steps \
    --learning_rate=5.0e-04 \
    --scale_lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --output_dir="${FINAL_DIR}"

CUDA_VISIBLE_DEVICES=$DEVICE accelerate launch --main_process_port $port scripts/textual_inversion.py \
    --pretrained_model_name_or_path=$MODEL_PATH \
    --train_data_dir=$nontar_dir \
    --learnable_property="object" \
    --placeholder_token="<statue>" \
    --initializer_token="statue" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=$finetune_steps \
    --learning_rate=5.0e-04 \
    --scale_lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --output_dir="${NONTAR_FINAL_DIR}"
