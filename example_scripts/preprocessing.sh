export DATA_DIR=YOUR_DATA_PATH
export RESULTS_DIR=YOUR_RESULTS_PATH

python src/data_preprocessing/compute_tisr_heppa.py \
    --tsir_data_path CMIP/CMIP6/inputs4MIP/historical/solar/solarforcing-ref-day_input4MIPs_solar_CMIP_SOLARIS-HEPPA-3-2_gn_18500101-22991231.nc \
    --output_dir CMIP/CMIP6/inputs4MIP/historical/solar/r64x32/1hr/tisr \
    --path_to_reference_file ERA5/weatherbench1/r64x32/2m_temperature/2m_temperature_2009_5.625deg.nc \
    --start_year 1979 \
    --end_year 2018 \
    --time_delta 1h \
    --start_month_day_hour 01-01T00

python src/data_preprocessing/compute_normalization.py \
    --input_dir ERA5/weatherbench1/r64x32/*/ \
    --start_year 1979 \
    --end_year 2008

python src/compute_climatology_metrics.py \
    --save_path dummy_experiment/climatology_forecast/evaluation/  \
    --forecast_length 3651d \
    --init_conditions_eval 2009-01-01T00