from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import six
import tensorflow as tf

from bioasq_biobert import optimization, modeling, tokenization, utils

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


def train_model():
    """
    Train the model.
    """
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    utils.validate_flags_or_throw(bert_config)
    tf.io.gfile.makedirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    train_examples = utils.read_squad_examples(input_file=FLAGS.train_file, is_training=True)
    num_train_steps = int(math.ceil(len(train_examples) / FLAGS.train_batch_size) * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Re-define run_config for training
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=int(num_train_steps / FLAGS.num_train_epochs),
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=int(num_train_steps / FLAGS.num_train_epochs),
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    rng = random.Random(12345)
    rng.shuffle(train_examples)

    model_fn = utils.model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    # We write to a temporary file to avoid storing very large constant tensors in memory.
    train_writer = utils.FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
        is_training=True)

    # Bootstrap
    for _ in range(int(FLAGS.num_train_epochs)):
        train_examples = utils.read_squad_examples(input_file=FLAGS.train_file, is_training=True)
        utils.convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length,
            is_training=True,
            output_fn=train_writer.process_feature)

    train_writer.close()
    tf.logging.info("***** Running training *****")
    tf.logging.info(f"Num orig examples = {len(train_examples) * int(FLAGS.num_train_epochs)}")
    tf.logging.info(f"Num split examples = {train_writer.num_features}")
    tf.logging.info(f"Num pos examples = {train_writer.num_pos_features}")
    tf.logging.info(f"Batch size = {FLAGS.train_batch_size}")
    tf.logging.info(f"Num steps = {num_train_steps}")
    tf.logging.info(f"Num warmup steps = {num_warmup_steps}")

    train_input_fn = utils.input_fn_builder(
        input_file=train_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)

    estimator.train(input_fn=train_input_fn, steps=num_train_steps)


def predict(
        estimator: tf.contrib.tpu.TPUEstimator,
        tokenizer: tokenization.FullTokenizer,
        docone_directory: str):
    """
    Use a given estimator to run predictions on a set of docone data files.

    :param estimator:
    :param tokenizer:
    :param docone_directory: Google cloud directory to a set of docone questions.
    """
    pdir = os.path.join(FLAGS.output_dir, 'predictions')
    if not tf.io.gfile.exists(pdir):
        tf.io.gfile.makedirs(pdir)

    do_iter = utils.DoconeQuestionIterator(docone_directory)
    for i, eval_examples in enumerate(do_iter.iterate()):
        if i > 0 and i % 100 == 0:
            tf.logging.info(f"Finished {i * 100000} examples")

        eval_writer = utils.FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
            is_training=False)
        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        utils.convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length,
            is_training=False,
            output_fn=append_feature)

        eval_writer.close()

        tf.logging.info("***** Running predictions *****")
        tf.logging.info(f"Num orig examples = {len(eval_examples)}")
        tf.logging.info(f"Num split examples = {len(eval_features)}")
        tf.logging.info(f"Batch size = {FLAGS.predict_batch_size}")

        predict_input_fn = utils.input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        # If running eval on the TPU, you will need to specify the number of steps
        all_results = []
        for result in estimator.predict(predict_input_fn, yield_single_examples=True):
            if len(all_results) % 100 == 0:
                tf.logging.info("Processing example: %d" % (len(all_results)))
            unique_id = int(result["unique_ids"])
            logits = [float(x) for x in result["logits"].flat]
            all_results.append(RawResult(unique_id=unique_id, logits=logits))

        output_prediction_file = os.path.join(pdir, f"predictions_{i}.json")
        output_nbest_file = os.path.join(pdir, f"nbest_predictions_{i}.json")
        output_null_log_odds_file = os.path.join(pdir, f"null_odds_{i}.json")

        utils.write_predictions(
            eval_examples, eval_features, all_results, FLAGS.n_best_size, FLAGS.max_answer_length,
            FLAGS.do_lower_case, output_prediction_file, output_nbest_file, output_null_log_odds_file)


def main(_):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    utils.validate_flags_or_throw(bert_config)
    tf.io.gfile.makedirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = utils.read_squad_examples(input_file=FLAGS.train_file, is_training=True)
        num_train_steps = int(math.ceil(len(train_examples) / FLAGS.train_batch_size) * FLAGS.num_train_epochs)
        # num_train_steps = FLAGS.save_checkpoints_steps
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        # Re-define run_config for training
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=FLAGS.master,
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=int(num_train_steps / FLAGS.num_train_epochs),
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=int(num_train_steps / FLAGS.num_train_epochs),
                num_shards=FLAGS.num_tpu_cores,
                per_host_input_for_training=is_per_host))

        rng = random.Random(12345)
        rng.shuffle(train_examples)

    model_fn = utils.model_fn_builder(
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
        train_writer = utils.FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
            is_training=True)

        # Bootstrap
        for epoch_idx in range(int(FLAGS.num_train_epochs)):
            train_examples = utils.read_squad_examples(input_file=FLAGS.train_file, is_training=True)

            utils.convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=FLAGS.max_seq_length,
                doc_stride=FLAGS.doc_stride,
                max_query_length=FLAGS.max_query_length,
                is_training=True,
                output_fn=train_writer.process_feature)

        train_writer.close()
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num orig examples = %d", len(train_examples) * int(FLAGS.num_train_epochs))
        tf.logging.info("  Num split examples = %d", train_writer.num_features)
        tf.logging.info("  Num pos examples = %d", train_writer.num_pos_features)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        tf.logging.info("  Num warmup steps = %d", num_warmup_steps)

        train_input_fn = utils.input_fn_builder(
            input_file=train_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)

        estimator.train(input_fn=train_input_fn, steps=num_train_steps)

    pdir = os.path.join(FLAGS.output_dir, 'predictions')
    if not tf.io.gfile.exists(pdir):
        tf.io.gfile.makedirs(pdir)

    if FLAGS.do_predict:
        do_iter = utils.DoconeQuestionIterator(FLAGS.docone_directory)
        for i, eval_examples in enumerate(do_iter.iterate()):
            if i > 0 and i % 100 == 0:
                tf.logging.info(f"Finished {i * 100000} examples")

            eval_writer = utils.FeatureWriter(
                filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
                is_training=False)
            eval_features = []

            def append_feature(feature):
                eval_features.append(feature)
                eval_writer.process_feature(feature)

            utils.convert_examples_to_features(
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

            predict_input_fn = utils.input_fn_builder(
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
                logits = [float(x) for x in result["logits"].flat]
                all_results.append(RawResult(unique_id=unique_id, logits=logits))

            output_prediction_file = os.path.join(pdir, f"predictions_{i}.json")
            output_nbest_file = os.path.join(pdir, f"nbest_predictions_{i}.json")
            output_null_log_odds_file = os.path.join(pdir, f"null_odds_{i}.json")

            utils.write_predictions(
                eval_examples, eval_features, all_results, FLAGS.n_best_size, FLAGS.max_answer_length,
                FLAGS.do_lower_case, output_prediction_file, output_nbest_file, output_null_log_odds_file)
