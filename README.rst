Tensorflow Sudoku Solver Benchmark
==================================

Benchmark tool evaluating training and inference from `sudoku-ml <https://github.com/cloudmercato/sudoku-ml>`_.


Install
-------

::

  pip install https://github.com/cloudmercato/sudoku-game/archive/refs/heads/master.zip
  pip install https://github.com/cloudmercato/sudoku-ml/archive/refs/heads/master.zip
  pip install https://github.com/cloudmercato/sudoku-ml-benchmark/archive/refs/heads/master.zip
  
Usage
-----

::

  usage: sudoku-ml-bench [-h] [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                         [--train-dataset-size TRAIN_DATASET_SIZE]
                         [--train-removed TRAIN_REMOVED]
                         [--infer-dataset-size INFER_DATASET_SIZE]
                         [--infer-removed INFER_REMOVED]
                         [--generator-processes GENERATOR_PROCESSES]
                         [--model-path MODEL_PATH]
                         [--model-load-file MODEL_LOAD_FILE]
                         [--model-save-file MODEL_SAVE_FILE] [--log-dir LOG_DIR]
                         [--tf-log-device] [--tf-dump-debug-info]
                         [--tf-profiler-port TF_PROFILER_PORT]
                         [--verbose VERBOSE] [--tf-verbose TF_VERBOSE]

  optional arguments:
    -h, --help            show this help message and exit
    --batch-size BATCH_SIZE
    --epochs EPOCHS
    --train-dataset-size TRAIN_DATASET_SIZE
    --train-removed TRAIN_REMOVED
    --infer-dataset-size INFER_DATASET_SIZE
    --infer-removed INFER_REMOVED
    --generator-processes GENERATOR_PROCESSES
    --model-path MODEL_PATH
                          Python path to the model to compile
    --model-load-file MODEL_LOAD_FILE
                          Model load file path (h5)
    --model-save-file MODEL_SAVE_FILE
                          Model save file path (h5)
    --log-dir LOG_DIR     Tensorboard log directory
    --tf-log-device       Determines whether TF compute device info is
                          displayed.
    --tf-dump-debug-info
    --tf-profiler-port TF_PROFILER_PORT
    --verbose VERBOSE, -v VERBOSE
    --tf-verbose TF_VERBOSE, -tfv TF_VERBOSE
    
Docker support
--------------

Dockerfile for classic Tensorflow and the GPU version are available: ::

  # For CPU
  docker build -f Dockerfile -t sudoku-ml-bench .
  docker run -it sudoku-ml-bench
  
  # For GPU
  docker build -f Dockerfile-gpu -t sudoku-ml-bench-gpu .
  docker run --gpus all --ipc=host -it sudoku-ml-bench-gpu
  # Add -e TF_CPP_MIN_LOG_LEVEL=3 to catch only the JSON output
  
The commands above will run a training, then save an inference. You can mount a volume on `/models/` to keep it. In the same idea you can mount a volume on `/log_dir/`, to retrive the Tensorboard data.
  
