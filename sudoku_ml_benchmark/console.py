import json
import time
import os
import sys
import argparse
import logging

import numpy as np
import tensorflow as tf

from sudoku_ml import datasets
from sudoku_ml import utils as ml_utils
from sudoku_ml import __version__ as ML_VERSION
from sudoku_ml_benchmark import utils
from sudoku_ml_benchmark import __version__ as VERSION


logger = logging.getLogger('sudoku_ml')
tf_logger = logging.getLogger('tensorflow')

PRECISON_POLICIES = (
    'mixed_float16',
    'mixed_bfloat16',
    'float16',
    'float32',
    'float64',
)

parser = argparse.ArgumentParser()
# Common
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--tpu', required=False, default=None)
parser.add_argument('--enable-xla', default=False, action="store_true")
parser.add_argument('--precision-policy', default=None, choices=PRECISON_POLICIES)
parser.add_argument('--optimize-model', default=False, action="store_true")
# Training
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--train-dataset-size', type=int, default=100000)
parser.add_argument('--train-removed', type=str, default='10,20')
# Inference
parser.add_argument('--infer-dataset-size', type=int, default=100000)
parser.add_argument('--infer-removed', type=str, default='10,20')
# Generator
parser.add_argument('--generator-processes', type=int, default=4)
parser.add_argument('--dataset-path', type=str, default=None)
# Model
parser.add_argument('--model-path', default='sudoku_ml.models.DEFAULT_MODEL',
                    help='Python path to the model to compile')
parser.add_argument('--model-load-file', default=None,
                    help='Model load file path (h5)')
parser.add_argument('--model-save-file', default='model.h5',
                    help='Model save file path (h5)')
# Log
parser.add_argument('--log-dir', default=None,
                    help='Tensorboard log directory')
parser.add_argument('--tf-log-device', default=False, action="store_true",
                    help='Determines whether TF compute device info is displayed.')
parser.add_argument('--tf-dump-debug-info', default=False, action="store_true")
parser.add_argument('--tf-profiler-port', default=0, type=int)
parser.add_argument('--verbose', '-v', default=3, type=int)
parser.add_argument('--tf-verbose', '-tfv', default=2, type=int)


def main():
    args = parser.parse_args()

    log_verbose = 60 - (args.verbose*10)
    log_handler = logging.StreamHandler()
    log_handler.setLevel(log_verbose)
    logger.addHandler(log_handler)
    logger.setLevel(log_verbose)

    tf_log_verbose = 60 - (args.tf_verbose*10)
    tf_logger.setLevel(tf_log_verbose)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', str(args.tf_verbose))

    logger.debug('Config: %s', vars(args))

    tf.debugging.set_log_device_placement(args.tf_log_device)
    if args.log_dir and args.tf_dump_debug_info:
        tf.debugging.experimental.enable_dump_debug_info(
            args.log_dir,
            tensor_debug_mode="FULL_HEALTH",
            circular_buffer_size=-1
        )
    if args.tf_profiler_port:
        tf.profiler.experimental.server.start(args.tf_profiler_port)

    if args.tpu is not None:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        scope = strategy.scope()
        scope.__enter__()

    if args.precision_policy:
        tf.keras.mixed_precision.set_global_policy(args.precision_policy)
    if args.enable_xla:
        tf.config.optimizer.set_jit(True)

    from sudoku_ml.agent import Agent
    agent = Agent(
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_path=args.model_path,
        model_load_file=args.model_load_file,
        model_save_file=args.model_save_file,
        log_dir=args.log_dir,
        verbose=args.tf_verbose,
    )
    if args.dataset_path:
        generator = datasets.FromFileGenerator(
            fd=args.dataset_path,
        )
    else:
        generator = datasets.Generator(
            processes=args.generator_processes,
        )

    output = {
        'version': VERSION,
        'ml_version': ML_VERSION,
        'py_version': sys.version,
        'np_version': np.version.full_version,
        'tf_version': tf.version.VERSION,
        # Training is optional
        'train_time': None,
        'train_speed': None,
        'train_start_loss': None,
        'train_end_loss': None,
        'train_dataset_gen_time': None,
    }
    output.update(vars(args))

    if not args.model_load_file:
        train_removed = ml_utils.parse_remove(args.train_removed)
        logger.info("Generating training dataset")
        start_time = time.time()
        train_dataset = generator.generate_training_dataset(
            count=args.train_dataset_size,
            removed=train_removed,
        )
        train_dataset_gen_time = time.time() - start_time
        logger.debug("Ended training dataset generation: %.2fsec", train_dataset_gen_time)

        logger.info("Start training")
        start_time = time.time()
        agent.train(
            runs=1,
            dataset=train_dataset
        )
        train_time = time.time() - start_time
        agent.save_model()
        del train_dataset
        output.update({
            'train_time': train_time,
            'train_speed': args.train_dataset_size / train_time,
            'train_start_loss': agent.model.history.history['loss'][0],
            'train_end_loss': agent.model.history.history['loss'][-1],
            'train_dataset_gen_time': train_dataset_gen_time,
        })

    infer_removed = ml_utils.parse_remove(args.infer_removed)
    logger.info("Generating inference dataset")
    start_time = time.time()
    infer_dataset = generator.generate_dataset(
        count=args.infer_dataset_size,
        removed=infer_removed,
    )
    infer_dataset_gen_time = time.time() - start_time
    logger.debug("Ended inference dataset generation: %.2fsec", infer_dataset_gen_time)

    valid_count = 0
    infer_times = []

    logger.info("Start inference")
    x, y = infer_dataset
    for i in range(args.infer_dataset_size):
        start_time = time.time()
        X, Y, value = agent.infer(x[i])
        infer_time = time.time() - start_time
        is_valid = y[i].reshape((9, 9))[X, Y] == value
        valid_count += is_valid
        infer_times.append(infer_time)
    infer_time = sum(infer_times)

    output.update({
        'infer_valid': valid_count,
        'infer_score': valid_count/args.infer_dataset_size,
        'infer_time': infer_time,
        'infer_speed': args.infer_dataset_size / infer_time,
        'infer_dataset_gen_time': infer_dataset_gen_time,
    })
    print(json.dumps(output, indent=2, cls=utils.NpEncoder))


if __name__ == "__main__":
    main()
