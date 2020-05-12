export BIOBERT_DIR=./model/bert_factoid
export BIOASQ_DIR=./datasets
export OUTPUT_DIR=./outputs/testing_factoid

python run_factoid.py \
     --do_train=True \
     --do_predict=True \
     --vocab_file=$BIOBERT_DIR/vocab.txt \
     --bert_config_file=$BIOBERT_DIR/bert_config.json \
     --init_checkpoint=$BIOBERT_DIR/model.ckpt-14599 \
     --max_seq_length=384 \
     --train_batch_size=12 \
     --learning_rate=5e-6 \
     --doc_stride=128 \
     --num_train_epochs=5.0 \
     --do_lower_case=False \
     --train_file=$BIOASQ_DIR/BioASQ-6b/train/Full-Abstract/BioASQ-train-factoid-6b-full-annotated.json \
     --predict_file=$BIOASQ_DIR/BioASQ-6b/test/Full-Abstract/BioASQ-test-factoid-6b-3.json \
     --output_dir=/tmp/factoid_output/
