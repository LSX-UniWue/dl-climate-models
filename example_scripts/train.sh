export DATA_DIR=YOUR_DATA_PATH
export RESULTS_DIR=YOUR_RESULTS_PATH

# EXP_NAME is the run name in WandB which is also used as the lowest dir in the output path for convenience
export RUN_NAME=dummy_run
export EXP_DIR=dummy_experiment/$RUN_NAME

python -u src/train.py --config example_configs/sfno_33vars.yaml \
      --model.net.init_args.embed_dim 64 \
      --model.net.init_args.num_layers 2 