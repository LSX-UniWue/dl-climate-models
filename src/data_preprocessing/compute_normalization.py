import argparse
import glob
import os
import time
import xarray as xr

from utils.data_utils import load_xr_dataset


def main(args):



    for input_dir in args.input_dirs:
        input_dir = os.path.join(os.environ['DATA_DIR'],input_dir)

        dir_paths = glob.glob(input_dir)
        print(f"Input directories: {dir_paths}")

        for path in dir_paths:
            if os.path.isdir(path):
                print(f"Processing {path}", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                
                out_dir = os.path.join(path, f"{args.output_dir}" , f"{args.start_year}_{args.end_year}")
                os.makedirs(out_dir, exist_ok=True)

                outpath = os.path.join(out_dir ,f'norm_values.nc')

                if os.path.exists(outpath) and not args.overwrite_existing:
                    print(f"Skipping {outpath} as it already exists")
                
                else:
                    files = glob.glob(os.path.join(path, '*.nc'))
                    dataset = load_xr_dataset(files,args.chunk_size)
                    print(dataset)


                    if 'time' in dataset.coords:
                        dataset = dataset.sel(time=slice(f"{args.start_year}-01-01",f"{args.end_year}-12-31"))
                        
                    mean = dataset.mean([v for v in dataset.dims if v != 'plev']).compute()
                    std = dataset.std([v for v in dataset.dims if v != 'plev']).compute()
                    mean = mean.rename_vars({v:f"mean_{v}" for v in mean.data_vars})
                    std = std.rename_vars({v:f"std_{v}" for v in std.data_vars})
                    normalize_data = xr.merge([mean,std])

                    print("Normalization data", normalize_data)
                    normalize_data.to_netcdf(outpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', type=str, help='Input file path', nargs='+')
    parser.add_argument('--output_dir', type=str, help='Output', default= 'normalization')
    parser.add_argument('--start_year', type=int, help='Start year', default= 1979)
    parser.add_argument('--end_year', type=int, help='End year', default= 2008)
    parser.add_argument('--chunk_size', type=int, help='Chunk size', default= 100)
    parser.add_argument('--overwrite_existing', type=bool, help='Overwrite existing files', default= False)


    args = parser.parse_args()


    main(args)