# !/bin/bash

GPU_ID=0
MODEL="runwayml/stable-diffusion-v1-5"
IMAGE_PATH="images/06.png"
CONTEXT_PROMPT="A painting of a woman in a blue dress playing a violin, alongside another woman in a red dress playing a piano behind the violinist, with women sitting in chairs, engrossed in listening to the music, in the style of {}"
OUTPUT_DIR="outputs/attention-mid0.25-js/attention06"
NEGATIVE_PROMPT="['woman', 'blue dress', 'violin', 'red dress', 'piano', 'women', 'chairs']"
DISTANGLE_LOSS_MID=0.25
DISTANGLE_LOSS_DOWN1=0.0
ATTN_LOSS="js"


function show_help {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -g, --gpu GPU_ID             GPU ID to use (default: $GPU_ID)"
    echo "  -m, --model MODEL            Model path (default: $MODEL)"
    echo "  -i, --image PATH             Training image path (default: $IMAGE_PATH)"
    echo "  -c, --context TEXT           Context prompt (default: $CONTEXT_PROMPT)"
    echo "  -o, --output DIR             Output directory (default: $OUTPUT_DIR)"
    echo "  -n, --negative TEXT          Negative prompt (default: $NEGATIVE_PROMPT)"
    echo "  --mid-loss VALUE             Mid disentangle loss weight (default: $DISTANGLE_LOSS_MID)"
    echo "  --down1-loss VALUE           Down1 disentangle loss weight (default: $DISTANGLE_LOSS_DOWN1)"
    echo "  -a, --attn-loss TYPE         Attention loss type (js|kl|cosine) (default: $ATTN_LOSS)"
    echo "  -h, --help                   Show this help message"
}

TEMP=$(getopt -o g:m:i:c:o:n:a:h --long gpu:,model:,image:,context:,output:,negative:,mid-loss:,down1-loss:,attn-loss:,help -n "$0" -- "$@")
if [ $? -ne 0 ]; then
    echo "Terminating..." >&2
    exit 1
fi

eval set -- "$TEMP"

while true; do
    case "$1" in
        -g|--gpu)
            GPU_ID="$2"; shift 2 ;;
        -m|--model)
            MODEL="$2"; shift 2 ;;
        -i|--image)
            IMAGE_PATH="$2"; shift 2 ;;
        -c|--context)
            CONTEXT_PROMPT="$2"; shift 2 ;;
        -o|--output)
            OUTPUT_DIR="$2"; shift 2 ;;
        -n|--negative)
            NEGATIVE_PROMPT="$2"; shift 2 ;;
        --mid-loss)
            DISTANGLE_LOSS_MID="$2"; shift 2 ;;
        --down1-loss)
            DISTANGLE_LOSS_DOWN1="$2"; shift 2 ;;
        -a|--attn-loss)
            ATTN_LOSS="$2"; shift 2 ;;
        -h|--help)
            show_help; exit 0 ;;
        --)
            shift; break ;;
        *)
            echo "Internal error!"; exit 1 ;;
    esac
done

if [[ "$ATTN_LOSS" != "js" && "$ATTN_LOSS" != "kl" && "$ATTN_LOSS" != "cosine" ]]; then
    echo "Error: Invalid attention loss type. Must be one of: js, kl, cosine"
    echo "Using default: js"
    ATTN_LOSS="js"
fi

NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" CUDA_VISIBLE_DEVICES=$GPU_ID python dreamstyler/train_attn.py \
    --pretrained_model_name_or_path "$MODEL" \
    --train_image_path "$IMAGE_PATH" \
    --context_prompt "$CONTEXT_PROMPT" \
    --output_dir "$OUTPUT_DIR" \
    --negative_prompt "$NEGATIVE_PROMPT" \
    --disentangle_loss_weight_mid "$DISTANGLE_LOSS_MID" \
    --disentangle_loss_weight_down1 "$DISTANGLE_LOSS_DOWN1" \
    --attn_loss "$ATTN_LOSS"
