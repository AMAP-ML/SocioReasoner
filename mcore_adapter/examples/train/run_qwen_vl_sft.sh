workdir=$(cd $(dirname $0); pwd)
parent_dir=$(dirname "$workdir")

WORLD_SIZE=8
TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=1

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
export DISABLE_VERSION_CHECK=1

USE_MCA=true

mca_options=" \
       --tensor_model_parallel_size ${TENSOR_MODEL_PARALLEL_SIZE} \
       --sequence_parallel \
       --pipeline_model_parallel_size ${PIPELINE_MODEL_PARALLEL_SIZE} \
       --bias_activation_fusion \
       --apply_rope_fusion \
       --use_distributed_optimizer" # \
       # --overlap_param_gather \ 
       # --overlap_grad_reduce" # projector may not training on some ranks, so overlap grad reduce not support

llama_factory_options=" \
       --deepspeed=${parent_dir}/config/ds_zero2.json"

options=" \
       --do_train \
       --stage=sft \
       --finetuning_type=full \
       --dataset_dir=$parent_dir/data \
       --dataset=pokemon_cap \
       --preprocessing_num_workers=8 \
       --cutoff_len=4096 \
       --template=qwen2_vl \
       --model_name_or_path=$MODEL_NAME \
       --output_dir=./tmp/ \
       --per_device_train_batch_size=1 \
       --gradient_accumulation_steps=2 \
       --num_train_epochs=2 \
       --learning_rate=2e-5 \
       --logging_steps=1 \
       --save_steps=100 \
       --lr_scheduler_type=cosine \
       --bf16"

if [ "$USE_MCA" = true ]; then
    options="$options $mca_options --use_mca"
else
    WORLD_SIZE=$(($WORLD_SIZE / $TENSOR_MODEL_PARALLEL_SIZE / $PIPELINE_MODEL_PARALLEL_SIZE))
    options="$options $llama_factory_options --use_mca=False"
fi

torchrun --nproc_per_node=$WORLD_SIZE $workdir/run_train.py $options
