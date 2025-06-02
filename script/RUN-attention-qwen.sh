# !/bin/bash

GPU_ID=0
MODEL="runwayml/stable-diffusion-v1-5"
# MODEL="stabilityai/stable-diffusion-2-1"

declare -A neg_prompts
while IFS=: read -r img neg; do
    img_trim=$(echo "$img" | xargs)
    neg_trim=$(echo "$neg" | xargs)
    neg_prompts["$img_trim"]="$neg_trim"
done < ./prompt/qwen_neg-complicated.txt

while IFS=, read -r img prompt; do
    idx=$(echo "$img" | grep -oE '^[0-9]+')
    img_path="./images/$img"
    context_prompt=$(echo "$prompt" | sed 's/^ //')
    placeholder_token="<attention${idx}>"
    output_dir="./outputs/attention-qwen-complicated-mid0.35-js/attention${idx}"

    neg_prompt="${neg_prompts[$img]}"

    echo "Processing image: $img_path"
    echo "Context prompt: $context_prompt"
    echo "Placeholder token: $placeholder_token"
    echo "Output directory: $output_dir"
    echo "Negative prompt: $neg_prompt"

    NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" CUDA_VISIBLE_DEVICES=$GPU_ID python dreamstyler/train_attn.py \
        --pretrained_model_name_or_path "$MODEL" \
        --train_image_path "$img_path" \
        --context_prompt "$context_prompt" \
        --placeholder_token "$placeholder_token" \
        --output_dir "$output_dir" \
        --negative_prompt "$neg_prompt" \
        --disentangle_loss_weight_mid 0.35 \
        --disentangle_loss_weight_down1 0.0 \
        --attn_loss "js" 
        # --visualize_mid_attn

    CUDA_VISIBLE_DEVICES=$GPU_ID python dreamstyler/inference_t2i.py \
        --sd_path "$MODEL" \
        --embedding_path "${output_dir}/embedding/final.bin" \
        --saveroot "${output_dir}/sample" \
        --placeholder_token "$placeholder_token"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python dreamstyler/inference_t2i.py \
        --sd_path "$MODEL" \
        --embedding_path "${output_dir}/embedding/final.bin" \
        --saveroot "${output_dir}/sample-con5.0" \
        --placeholder_token "$placeholder_token" \
        --con_gamma 5.0

done < ./prompt/qwen_caption-complicated.txt
