export DATA_DIR=YOUR_DATA_PATH
export RESULTS_DIR=YOUR_RESULTS_PATH

SEARCH_NAME=dummy_hyperparameter_search

FORECAST_TIME=3651d

seeds=(9982 6153 3909 5707 597)

# params that change the cuda memory usage
# batch_size, gradient_accumulation, n_steps, n_layers, embed_dim
params=(
    "64 1 1 4 128"
    "64 1 1 6 256"
    "64 1 1 8 512"
    "64 1 2 4 128"
    "32 2 2 6 256"
    "32 2 2 8 512"
    "32 2 4 4 128"
    "32 2 4 6 256"
    "16 4 4 8 512"
)


for tuple in "${params[@]}"; do
    set -- $tuple
    bs=$1
    ga=$2
    steps=$3
    layers=$4
    dim=$5
    for seed in "${seeds[@]}"; do
        export RUN_NAME=climax_8vars_steps$steps"_layers"$layers"_dim"$dim"_seed"$seed
        export EXP_DIR=$SEARCH_NAME/$RUN_NAME


        python -u src/train.py --config example_configs/climax_8vars.yaml \
            --seed_everything $seed \
            --data.batch_size $bs \
            --trainer.accumulate_grad_batches $ga \
            --data.train_config.n_steps $steps \
            --model.net.init_args.num_layers $layers \
            --model.net.init_args.embed_dim $dim

        python src/autoregressive_rollout.py \
            --config_paths $EXP_DIR/config.yaml \
            --forecast_time $FORECAST_TIME \
            --init_dates 2009-01-01T00 2009-02-01T00

        python src/evaluation/evaluate_run.py \
            --run_path $EXP_DIR \
            --eval_folder evaluation/$FORECAST_TIME/ \
            --ref_dataset ERA5/weatherbench1/r64x32/*/ \
            --normalization_path ERA5/weatherbench1/r64x32/*/normalization/1979_2008
    done
done


python src/evaluation/plot_metrics.py \
    --run_path $SEARCH_NAME/* \
    --exp_name "evaluation/$FORECAST_TIME/2009-01-01 00:00:00"

python src/evaluation/plots_per_run.py \
    --run_path $SEARCH_NAME/* \
    --exp_name "evaluation/$FORECAST_TIME/2009-01-01 00:00:00" \