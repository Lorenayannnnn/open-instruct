export PYTHONPATH=:${PYTHONPATH}

# open-instruct/docs/archived_dev_scripts/olmo2_1124.sh
# python mason.py \
#     --cluster ai2/jupiter --image nathanl/open_instruct_auto --pure_docker_mode \
#     --workspace ai2/tulu-3-dev \
#     --priority urgent \
#     --preemptible \
#     --num_nodes 4 \
#     --image costah/open_instruct_ppo_olmo23 \
#     --budget ai2/jupiter \
#     --gpus 8 -- pip install git+https://github.com/vwxyzjn/transformers.git@olmo2-classification \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/ppo_vllm_thread_ray_gtrl_olmo.py \
#     --beta $beta \
#     --learning_rate $lr \
#     --exp_name $exp_name \
#     --seed $seed \
#     --output_dir "/weka/oe-adapt-default/costah/models/olmo2/${exp_name}" \
#     --save_freq 60 \
#     --try_launch_beaker_eval_jobs_on_weka \
#     --hf_metadata_dataset allenai/olmo-instruct-evals \
#     --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
#     --dataset_mixer_list_splits train \
#     --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
#     --dataset_mixer_eval_list_splits train \
#     --max_token_length 2048 \
#     --max_prompt_token_length 2048 \
#     --response_length 2048 \
#     --model_name_or_path allenai/open_instruct_dev \
#     --model_revision 1208_dpo_13b_tune8e-7__allenai_open_instruct_dev__7__1733807564 \
#     --reward_model_path allenai/open_instruct_dev \
#     --reward_model_revision reward_modeling__1__1733807985 \
#     --non_stop_penalty \
#     --stop_token eos \
#     --temperature 1.0 \
#     --chat_template_name tulu \
#     --total_episodes 200000 \
#     --penalty_reward_value -10.0 \
#     --deepspeed_stage 3 \
#     --per_device_train_batch_size 2 \
#     --local_rollout_forward_batch_size 2 \
#     --local_mini_batch_size 8 \
#     --local_rollout_batch_size 8 \
#     --actor_num_gpus_per_node 7 8 8 8 \
#     --vllm_tensor_parallel_size 1 \
#     --apply_verifiable_reward true \
#     --num_evals 3 \
#     --reward_model_multiplier 0.0 \
#     --gradient_checkpointing \
#     --with_tracking

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/OLMo-2-1124-13B-DPO \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --chat_template_name tulu \
    --add_bos \
    --learning_rate 3e-7 \
    --total_episodes 400000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 8 \
    --local_rollout_batch_size 8 \
    --actor_num_gpus_per_node 7 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.9 \
    --beta 0.1 \
    --apply_verifiable_reward true \
    --output_dir output/olmo2_rlvr_13b \
    --seed 52 \
    --num_evals 3 \
    --save_freq 240 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking

# allenai/OLMo-2-0425-1B-DPO
# allenai/OLMo-2-1124-7B-DPO
# CUDA_VISIBLE_DEVICES=5,6,7 python open_instruct/ppo_vllm_thread_ray_gtrl.py \
#     --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
#     --dataset_mixer_list_splits train \
#     --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
#     --dataset_mixer_eval_list_splits train \
#     --max_token_length 2048 \
#     --max_prompt_token_length 2048 \
#     --response_length 2048 \
#     --model_name_or_path allenai/OLMo-2-0425-1B-DPO \
#     --non_stop_penalty \
#     --stop_token eos \
#     --temperature 1.0 \
#     --chat_template_name tulu \
#     --add_bos \
#     --learning_rate 3e-7 \
#     --total_episodes 96 \
#     --penalty_reward_value -10.0 \
#     --deepspeed_stage 3 \
#     --per_device_train_batch_size 2 \
#     --local_rollout_forward_batch_size 2 \
#     --local_mini_batch_size 8 \
#     --local_rollout_batch_size 8 \
#     --actor_num_gpus_per_node 2 \
#     --vllm_tensor_parallel_size 1 \
#     --vllm_gpu_memory_utilization 0.9 \
#     --beta 0.1 \
#     --apply_verifiable_reward true \
#     --output_dir output/test_olmo2_rlvr_7b \
#     --seed 52 \
#     --num_evals 3 \
#     --save_freq 3 \
#     --reward_model_multiplier 0.0 \
#     --gradient_checkpointing \
#     --with_tracking 


# --per_device_train_batch_size 4 \
# --local_rollout_forward_batch_size 4 \
#bash scripts/train/rlvr/olmo2_rlvr.sh