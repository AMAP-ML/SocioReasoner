workdir=$(cd $(dirname $0); pwd)
parent_dir=$(dirname "$workdir")

WORLD_SIZE=8
TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=1

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

USE_MCA=true

export DISABLE_VERSION_CHECK=1

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
       --stage=dpo \
       --pref_loss=sigmoid \
       --finetuning_type=full \
       --dataset_dir=$parent_dir/data \
       --dataset=ultrafeedback \
       --preprocessing_num_workers=8 \
       --cutoff_len=8192 \
       --template=llama3 \
       --model_name_or_path=$MODEL_NAME \
       --output_dir=./tmp/ \
       --per_device_train_batch_size=1 \
       --gradient_accumulation_steps=8 \
       --num_train_epochs=2 \
       --learning_rate=2e-6 \
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
