workdir=$(cd $(dirname $0); pwd)
parent_dir=$(dirname "$workdir")

WORLD_SIZE=8
TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=1

MODEL_NAME="Qwen/Qwen3-8B"

export DISABLE_VERSION_CHECK=1

USE_MCA=true

mca_options=" \
       --tensor_model_parallel_size ${TENSOR_MODEL_PARALLEL_SIZE} \
       --sequence_parallel \
       --pipeline_model_parallel_size ${PIPELINE_MODEL_PARALLEL_SIZE} \
       --use_distributed_optimizer \
       --bias_activation_fusion \
       --apply_rope_fusion \
       --overlap_param_gather \
       --overlap_grad_reduce"

llama_factory_options=" \
       --deepspeed=${parent_dir}/config/ds_zero2.json"

options=" \
       --do_train \
       --stage=sft \
       --finetuning_type=full \
       --dataset_dir=$parent_dir/data \
       --dataset=belle_2m \
       --preprocessing_num_workers=8 \
       --cutoff_len=8192 \
       --template=qwen3 \
       --model_name_or_path=$MODEL_NAME \
       --output_dir=./tmp/ \
       --per_device_train_batch_size=1 \
       --gradient_accumulation_steps=4 \
       --calculate_per_token_loss=True \
       --max_steps=100 \
       --learning_rate=2e-5 \
       --logging_steps=1 \
       --save_steps=50 \
       --lr_scheduler_type=cosine \
       --bf16"

if [ "$USE_MCA" = true ]; then
    options="$options $mca_options --use_mca"
else
    WORLD_SIZE=$(($WORLD_SIZE / $TENSOR_MODEL_PARALLEL_SIZE / $PIPELINE_MODEL_PARALLEL_SIZE))
    options="$options $llama_factory_options --use_mca=False"
fi

torchrun --nproc_per_node=$WORLD_SIZE $workdir/run_train.py $options
