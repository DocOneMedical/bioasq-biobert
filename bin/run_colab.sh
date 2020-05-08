export BASE_DIR=gs://biobert_params/biobert_apr
export BIOBERT_DIR=${BASE_DIR}/bert_yesno
export BIOASQ_DIR=${BASE_DIR}/datasets

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
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

export OUTPUT_DIR=${BASE_DIR}/outputs/${out}

python run_yesno.py \
    --do_train=${do_train} \
    --do_predict=True \
    --docone --docone_directory=$BIOASQ_DIR/D000006_D001145 \
    --vocab_file=$BIOBERT_DIR/vocab.txt \
    --bert_config_file=$BIOBERT_DIR/bert_config.json \
    --init_checkpoint=$BIOBERT_DIR/model.ckpt-14470 \
    --max_seq_length=${max_seq} \
    --train_batch_size=16 \
    --learning_rate=5e-6 \
    --doc_stride=128 \
    --do_lower_case=False \
    --num_train_epochs=2 \
    --train_file=$BIOASQ_DIR/BioASQ-6b/train/Snippet-as-is/BioASQ-train-yesno-6b-snippet.json \
    --predict_file=$BIOASQ_DIR/BioASQ-6b/test/Snippet-as-is/BioASQ-test-yesno-6b-3-snippet.json \
    --output_dir=${OUTPUT_DIR} \
    --use_tpu --tpu_name=grpc://${tpu_ip}:8470 
