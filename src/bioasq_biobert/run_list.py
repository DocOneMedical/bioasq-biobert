# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from datetime import datetime
import os
import random
import tensorflow as tf

from bioasq_biobert import modeling, tokenization
from bioasq_biobert.utils import DoconeQuestionIterator, read_squad_examples, \
    convert_examples_to_features, model_fn_builder, input_fn_builder, write_predictions, FeatureWriter, \
    validate_flags_or_throw

flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string("bert_config_file", None,
                    "The config json file corresponding to the pre-trained BERT model. "
                    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("output_dir", None,
                    "The output directory where the model checkpoints will be written.")

# Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string("predict_file", None,
                    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string("init_checkpoint", None,
                    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_lower_case", True,
                  "Whether to lower case the input text. Should be True for uncased "
                  "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 384,
                     "The maximum total input sequence length after WordPiece tokenization. "
                     "Sequences longer than this will be truncated, and sequences shorter "
                     "than this will be padded.")

flags.DEFINE_integer("doc_stride", 128,
                     "When splitting up a long document into chunks, how much stride to "
                     "take between chunks.")

flags.DEFINE_integer("max_query_length", 64,
                     "The maximum number of tokens for the question. Questions longer than "
                     "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 12, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 1e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 5.0, "Total number of training epochs to perform.")

flags.DEFINE_float("warmup_proportion", 0.1,
                   "Proportion of training to perform linear learning rate warmup for. "
                   "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("n_best_size", 20,
                     "The total number of n-best predictions to generate in the "
                     "nbest_predictions.json output file.")

flags.DEFINE_integer("max_answer_length", 30,
                     "The maximum length of an answer that can be generated. This is needed "
                     "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string("tpu_name", None,
                       "The Cloud TPU to use for training. This should be either the name "
                       "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
                       "url.")

tf.flags.DEFINE_string("tpu_zone", None,
                       "[Optional] GCE zone where the Cloud TPU is located in. If not "
                       "specified, we will attempt to automatically detect the GCE project from "
                       "metadata.")

tf.flags.DEFINE_string("gcp_project", None,
                       "[Optional] Project name for the Cloud TPU-enabled project. If not "
                       "specified, we will attempt to automatically detect the GCE project from "
                       "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer("num_tpu_cores", 8,
                     "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool("verbose_logging", False,
                  "If true, all of the warnings related to data processing will be printed. "
                  "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool("version_2_with_negative", False,
                  "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float("null_score_diff_threshold", 0.0,
                   "If null_score - best_non_null is greater than the threshold predict null.")

flags.DEFINE_integer("input_shuffle_seed", 12345, "")

# Docone additions
flags.DEFINE_bool("docone", False, "If true, use the docone set for prediction.")
flags.DEFINE_string("docone_directory", None, "SQuAD json for training. E.g., train-v1.1.json")
flags.DEFINE_integer("docone_chunk", None, "chunk to start at")

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])


def main(_):
    tf.logging.set_verbosity(tf.logging.WARN)

    start_t = datetime.now()
    print(start_t)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    validate_flags_or_throw(bert_config)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=FLAGS.master,
            model_dir=FLAGS.output_dir,
            session_config=session_config,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            keep_checkpoint_max=50,
            tpu_config=tf.contrib.tpu.TPUConfig(
                    iterations_per_loop=FLAGS.iterations_per_loop,
                    num_shards=FLAGS.num_tpu_cores,
                    per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = read_squad_examples(
                input_file=FLAGS.train_file, is_training=True)
        num_train_steps = int(
                len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)

        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        # Pre-shuffle the input to avoid having to make a very large shuffle
        # buffer in in the `input_fn`.
        rng = random.Random(FLAGS.input_shuffle_seed)
        rng.shuffle(train_examples)
        print('#train_examples', len(train_examples))
        print('#train_steps', num_train_steps)

    model_fn = model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        # We write to a temporary file to avoid storing very large constant tensors
        # in memory.
        train_writer = FeatureWriter(
                filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
                is_training=True)
        convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=FLAGS.max_seq_length,
                doc_stride=FLAGS.doc_stride,
                max_query_length=FLAGS.max_query_length,
                is_training=True,
                output_fn=train_writer.process_feature)
        train_writer.close()

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num orig examples = %d", len(train_examples))
        tf.logging.info("  Num split examples = %d", train_writer.num_features)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        del train_examples

        train_input_fn = input_fn_builder(
                input_file=train_writer.filename,
                seq_length=FLAGS.max_seq_length,
                is_training=True,
                drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    pdir = os.path.join(FLAGS.output_dir, 'predictions')
    if not tf.io.gfile.exists(pdir):
        tf.io.gfile.makedirs(pdir)

    if FLAGS.do_predict:
        if FLAGS.docone:
            do_iter = DoconeQuestionIterator(FLAGS.docone_directory)
            for i, eval_examples in enumerate(do_iter.iterate()):
                tf.logging.warning(f"Processing chunk {i}".format(i))

                if FLAGS.docone_chunk and i < FLAGS.docone_chunk:
                    continue

                eval_writer = FeatureWriter(
                    filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
                    is_training=False)
                eval_features = []
                
                def append_feature(feature):
                    eval_features.append(feature)
                    eval_writer.process_feature(feature)

                convert_examples_to_features(
                    examples=eval_examples,
                    tokenizer=tokenizer,
                    max_seq_length=FLAGS.max_seq_length,
                    doc_stride=FLAGS.doc_stride,
                    max_query_length=FLAGS.max_query_length,
                    is_training=False,
                    output_fn=append_feature)
                eval_writer.close()

                tf.logging.info("***** Running predictions *****")
                tf.logging.info("  Num orig examples = %d", len(eval_examples))
                tf.logging.info("  Num split examples = %d", len(eval_features))
                tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

                all_results = []

                predict_input_fn = input_fn_builder(
                    input_file=eval_writer.filename,
                    seq_length=FLAGS.max_seq_length,
                    is_training=False,
                    drop_remainder=False)

                # If running eval on the TPU, you will need to specify the number of
                # steps.
                all_results = []
                for result in estimator.predict(predict_input_fn, yield_single_examples=True):
                    if len(all_results) % 100 == 0:
                        tf.logging.info("Processing example: %d" % (len(all_results)))
                    unique_id = int(result["unique_ids"])
                    start_logits = [float(x) for x in result["start_logits"].flat]
                    end_logits = [float(x) for x in result["end_logits"].flat]
                    all_results.append(
                        RawResult(
                            unique_id=unique_id,
                            start_logits=start_logits,
                            end_logits=end_logits
                        ))

                output_prediction_file = os.path.join(pdir, f"predictions_{i}.json")
                output_nbest_file = os.path.join(pdir, f"nbest_predictions_{i}.json")
                output_null_log_odds_file = os.path.join(pdir, f"null_odds_{i}.json")

                write_predictions(eval_examples, eval_features, all_results,
                                  FLAGS.n_best_size, FLAGS.max_answer_length,
                                  FLAGS.do_lower_case, output_prediction_file,
                                  output_nbest_file, output_null_log_odds_file)

        else:
            eval_examples = read_squad_examples(
                    input_file=FLAGS.predict_file, is_training=False)

            eval_writer = FeatureWriter(
                    filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
                    is_training=False)
            eval_features = []

            def append_feature(feature):
                eval_features.append(feature)
                eval_writer.process_feature(feature)

            convert_examples_to_features(
                    examples=eval_examples,
                    tokenizer=tokenizer,
                    max_seq_length=FLAGS.max_seq_length,
                    doc_stride=FLAGS.doc_stride,
                    max_query_length=FLAGS.max_query_length,
                    is_training=False,
                    output_fn=append_feature)

            eval_writer.close()

            tf.logging.info("***** Running predictions *****")
            tf.logging.info("  Num orig examples = %d", len(eval_examples))
            tf.logging.info("  Num split examples = %d", len(eval_features))
            tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

            predict_input_fn = input_fn_builder(
                    input_file=eval_writer.filename,
                    seq_length=FLAGS.max_seq_length,
                    is_training=False,
                    drop_remainder=False)

            # If running eval on the TPU, you will need to specify the number of
            # steps.
            all_results = []
            for result in estimator.predict(
                    predict_input_fn, yield_single_examples=True):
                if len(all_results) % 1000 == 0:
                    tf.logging.info("Processing example: %d" % (len(all_results)))
                unique_id = int(result["unique_ids"])
                start_logits = [float(x) for x in result["start_logits"].flat]
                end_logits = [float(x) for x in result["end_logits"].flat]
                all_results.append(
                    RawResult(
                        unique_id=unique_id,
                        start_logits=start_logits,
                        end_logits=end_logits
                    ))

            """
            # example: BioASQ-test-list-6b-1.json
            # example: BioASQ-test-list-7b-3-snippet.json
            # example: BioASQ-test-list-6b-snippet-all.json
            pred_filename = os.path.split(FLAGS.predict_file)[1]
            pred_filename = pred_filename.replace('BioASQ-test-list-', '')
            pred_filename = pred_filename.replace('.json', '')
            if '-all' == pred_filename[-4:]:
                    task_batch = pred_filename[0] + 'B'  # no batch number
                    snp = pred_filename[2:-4]
                    golden = '/hdd1/bioasq/{}b_test/{}_total_golden.json'.format(
                            pred_filename[0], task_batch)
            else:
                    task_batch = pred_filename.replace('b-', 'B')
                    snp = task_batch[3:]
                    task_batch = task_batch[:3]
                    golden = '/hdd1/biobert/BioASQ/{}_golden.json'.format(task_batch)

            output_prediction_file = os.path.join(
                    FLAGS.output_dir, "{}_predictions.json".format(task_batch))
            output_nbest_file = os.path.join(
                    FLAGS.output_dir, "{}_nbest_predictions.json".format(task_batch))
            output_null_log_odds_file = os.path.join(
                    FLAGS.output_dir, "{}_null_odds.json".format(task_batch))
            """
            output_prediction_file = os.path.join(
                    FLAGS.output_dir, "predictions.json")
            output_nbest_file = os.path.join(
                    FLAGS.output_dir, "nbest_predictions.json")
            output_null_log_odds_file = os.path.join(
                    FLAGS.output_dir, "null_odds.json")

            write_predictions(eval_examples, eval_features, all_results,
                              FLAGS.n_best_size, FLAGS.max_answer_length,
                              FLAGS.do_lower_case, output_prediction_file,
                              output_nbest_file, output_null_log_odds_file)

            """
            # convert
            print("\nConvert to BioASQ format")
            import subprocess
            outdir = FLAGS.output_dir[:-1] if FLAGS.output_dir[-1] == '/' \
                    else FLAGS.output_dir
            out_file = '{}_list_result{}.json'.format(task_batch, snp)
            print('BioASQ format output', os.path.join(outdir, out_file))

            eval_proc = subprocess.Popen(
                    ['python3', './biocodes/transform_n2b_list.py',
                     '--nbest_path={}/{}_nbest_predictions.json'.format(outdir, task_batch),
                     # '-s',
                     '--output_path=' + outdir,
                     '--output_file=' + out_file],
                    cwd='.',
                    stdout=subprocess.PIPE
            )
            stdout, stderr = eval_proc.communicate()
            print(stdout.decode('utf-8'))
            if stderr is not None:
                    print(stderr.decode('utf-8'))

            print("\nEvaluation")
            # https://github.com/BioASQ/Evaluation-Measures/blob/master/flat/BioASQEvaluation/src/evaluation/EvaluatorTask1b.java#L59

            print('pred  ', os.path.join(outdir, out_file))
            print('golden', golden)

            if os.path.exists(golden):
                    # 1: [1, 2],    3: [3, 4],  5: [5, 6]
                    task_e = {
                            '1': 1,
                            '2': 1,
                            '3': 3,
                            '4': 3,
                            '5': 5,
                            '6': 5,
                    }

                    evalproc1 = subprocess.Popen(
                            ['java', '-Xmx10G', '-cp',
                             '$CLASSPATH:./flat/BioASQEvaluation/dist/BioASQEvaluation.jar',
                             'evaluation.EvaluatorTask1b', '-phaseB',
                             '-e', '{}'.format(task_e[task_batch[0]]),
                             golden, os.path.join(outdir, out_file)],
                            cwd='/hdd1/biobert/Evaluation-Measures',
                            stdout=subprocess.PIPE
                    )
                    stdout, _ = evalproc1.communicate()
                    print('\t'.join(
                                    ['{:.4f}'.format(float(v))
                                     for v in stdout.decode('utf-8').split(' ')][4:7]), sep='\t')
                    """
    print('\nElapsed time', datetime.now() - start_t)


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
