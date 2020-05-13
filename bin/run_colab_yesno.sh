export BASE_DIR=gs://biobert_params/biobert_apr

while getopts ":o:t:m:d:" opt; do
  case $opt in
    o) out="$OPTARG"
    ;;
    t) tpu_ip="$OPTARG"
    ;;
    m) max_seq="$OPTARG"
    ;;
    d) do_train="$OPTARG"
    ;;
    k) dataset="$OPTARG"
    ;;
    c) checkpt="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

export DATASET_DIR=${BASE_DIR}/datasets
export COMMON_DIR=${BASE_DIR}/common
export OUTPUT_DIR=${BASE_DIR}/outputs/${out}

python run_yesno.py \
    --do_train=${do_train} \
    --do_predict=True \
    --docone --docone_directory=${DATASET_DIR}/${dataset} \
    --vocab_file=${COMMON_DIR}/vocab.txt \
    --bert_config_file=${COMMON_DIR}/bert_config.json \
    --init_checkpoint=${BASE_DIR}/${checkpt} \
    --max_seq_length=${max_seq} \
    --train_batch_size=16 \
    --learning_rate=5e-6 \
    --doc_stride=128 \
    --do_lower_case=False \
    --num_train_epochs=2 \
    --train_file=${DATASET_DIR}/train/Snippet-as-is/BioASQ-train-yesno-6b-snippet.json \
    --predict_file=${DATASET_DIR}/test/Snippet-as-is/BioASQ-test-yesno-6b-3-snippet.json \
    --output_dir=${OUTPUT_DIR} \
    --use_tpu --tpu_name=grpc://${tpu_ip}:8470 
