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


python run_list.py \
     --do_train=${do_train} \
     --do_predict=True \
     --vocab_file=${COMMON_DIR}/vocab.txt \
     --docone --docone_directory=${DATASET_DIR}/${dataset} \
     --bert_config_file=${COMMON_DIR}/bert_config.json \
     --init_checkpoint=${BASE_DIR}/${checkpt} \
     --max_seq_length=${max_seq} \
     --train_batch_size=10 \
     --learning_rate=5e-6 \
     --doc_stride=128 \
     --num_train_epochs=7 \
     --do_lower_case=False \
     --train_file=${DATASET_DIR}/BioASQ-6b/train/Full-Abstract/BioASQ-train-list-6b-full-annotated.json \
     --predict_file=${DATASET_DIR}/BioASQ-6b/test/Full-Abstract/BioASQ-test-list-6b-3.json \
     --output_dir=${OUTPUT_DIR}

# --tpu --tpu_ip=???
