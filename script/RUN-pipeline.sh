# !/bin/bash

GPU_ID=0
MODEL="runwayml/stable-diffusion-v1-5"
EMBEDDING_PATH="outputs/attention-mid0.25-js/attention06/embedding/final.bin"
OUTPUT_DIR="outputs/attention-mid0.25-js/attention06"
NUM_SAMPLES_BEFORE=1
NUM_SAMPLES_AFTER=5
STYLE_IMAGE="images/06.png"
PROMPT="a painting of a dog"
PROMPT_LOGGER="prompt"
HISTORY_LOGGER="history"

function show_help {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -g, --gpu GPU_ID             GPU ID to use (default: $GPU_ID)"
    echo "  -m, --model MODEL            Model path (default: $MODEL)"
    echo "  -e, --embedding PATH         Embedding path (default: $EMBEDDING_PATH)"
    echo "  -o, --output DIR             Output directory (default: $OUTPUT_DIR)"
    echo "  -b, --samples-before NUM     Number of samples before debate (default: $NUM_SAMPLES_BEFORE)"
    echo "  -a, --samples-after NUM      Number of samples after debate (default: $NUM_SAMPLES_AFTER)"
    echo "  -s, --style-image PATH       Style image path (default: $STYLE_IMAGE)"
    echo "  -p, --prompt TEXT            Prompt text (default: $PROMPT)"
    echo "  --prompt-logger FILENAME     Prompt logger filename (default: $PROMPT_LOGGER)"
    echo "  --history-logger FILENAME    History logger filename (default: $HISTORY_LOGGER)"
    echo "  -h, --help                   Show this help message"
}

TEMP=$(getopt -o g:m:e:o:b:a:s:p:h --long gpu:,model:,embedding:,output:,samples-before:,samples-after:,style-image:,prompt:,prompt-logger:,history-logger:,help -n "$0" -- "$@")
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
        -e|--embedding)
            EMBEDDING_PATH="$2"; shift 2 ;;
        -o|--output)
            OUTPUT_DIR="$2"; shift 2 ;;
        -b|--samples-before)
            NUM_SAMPLES_BEFORE="$2"; shift 2 ;;
        -a|--samples-after)
            NUM_SAMPLES_AFTER="$2"; shift 2 ;;
        -s|--style-image)
            STYLE_IMAGE="$2"; shift 2 ;;
        -p|--prompt)
            PROMPT="$2"; shift 2 ;;
        --prompt-logger)
            PROMPT_LOGGER="$2"; shift 2 ;;
        --history-logger)
            HISTORY_LOGGER="$2"; shift 2 ;;
        -h|--help)
            show_help; exit 0 ;;
        --)
            shift; break ;;
        *)
            echo "Internal error!"; exit 1 ;;
    esac
done


mkdir -p "$OUTPUT_DIR"

echo "========== Inference and Debate Pipeline =========="
echo "========== Inference =========="
CUDA_VISIBLE_DEVICES=$GPU_ID python dreamstyler/inference_t2i.py \
    --sd_path "$MODEL" \
    --embedding_path "$EMBEDDING_PATH" \
    --saveroot "$OUTPUT_DIR/before_debate" \
    --prefix "before_debate" \
    --num_samples "$NUM_SAMPLES_BEFORE" \
    --prompt "${PROMPT}, in the style of {}" \

echo "========== Debate =========="
for i in $(seq 0 $(($NUM_SAMPLES_BEFORE - 1))); do
    IMAGE_NAME="before_debate_$i.png"
    INPUT_IMG="$OUTPUT_DIR/before_debate/$IMAGE_NAME"

    echo "Debating image: $INPUT_IMG"
    CUDA_VISIBLE_DEVICES=$GPU_ID python agents/debate_image_rounds.py \
        --generated-img "$INPUT_IMG" \
        --style-img "$STYLE_IMAGE" \
        --prompt "$PROMPT" \
        --prompt-logger "$OUTPUT_DIR/$PROMPT_LOGGER" \
        --history-logger "$OUTPUT_DIR/$HISTORY_LOGGER"

    SUGGESTED_DESC=$(tail -n 1 "$OUTPUT_DIR/$PROMPT_LOGGER" | awk -F'"' '{print $2}')
    if [ -z "$SUGGESTED_DESC" ]; then
        SUGGESTED_DESC=$(tail -n 1 "$OUTPUT_DIR/$PROMPT_LOGGER" | awk -F',' '{print $4}' | awk '{$1=$1;print}')
    fi
    SUGGESTED_DESC=$(echo "$SUGGESTED_DESC" | sed 's/\.$//')
    SUGGESTED_DESC="${SUGGESTED_DESC}, in the style of {}"
    echo "Suggested description: $SUGGESTED_DESC"
    echo "Inferencing image: $INPUT_IMG"
    CUDA_VISIBLE_DEVICES=$GPU_ID python dreamstyler/inference_t2i.py \
    --sd_path "$MODEL" \
    --embedding_path "$EMBEDDING_PATH" \
    --saveroot "$OUTPUT_DIR/after_debate" \
    --prefix "${IMAGE_NAME}_after_debate" \
    --num_samples "$NUM_SAMPLES_AFTER" \
    --prompt "$SUGGESTED_DESC" \

done