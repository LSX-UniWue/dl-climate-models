export DATA_DIR=YOUR_DATA_PATH
export RESULTS_DIR=YOUR_RESULTS_PATH

FORECAST_TIME=3d

python src/autoregressive_rollout.py \
    --config_paths dummy_experiment/dummy_run/config.yaml \
    --forecast_time $FORECAST_TIME \
    --init_dates 2009-01-01T00 2009-02-01T00

python src/evaluation/evaluate_run.py \
    --run_path dummy_experiment/dummy_run/ \
    --eval_folder evaluation/$FORECAST_TIME/ \
    --ref_dataset ERA5/weatherbench1/r64x32/*/ \
    --normalization_path ERA5/weatherbench1/r64x32/*/normalization/1979_2008
   