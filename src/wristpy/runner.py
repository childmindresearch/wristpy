import json
import warnings
from argparse import ArgumentParser

import wristpy
from wristpy.common import data_model
from wristpy.ggir import calibration, compare_dataframes, metrics_calc
from wristpy.io.loaders import gt3x

warnings.filterwarnings('ignore')

def load_config(filepath: str)->dict:
    """Loads configurations for file paths to gt3x raw file, and ggir output file.

    Args:
        filepath: file path to config.json file.

    Returns:
        dictionary with file paths to use for finding gt3x and ggir files.
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f'Config file: {filepath} not found, are you in the right directory?')

def update_config(filepath: str, gt3x_path: str, ggir_path: str)-> None:
    """Writes new filepaths input by users to search for gt3x file and ggir output file.

    Args:
        filepath: file path to config.json file.
        gt3x_path: new path to gt3x file.
        ggir_path: new path to ggir output file.
    
    Returns:
        None.
    """
    new_config = {
        "gt3x_raw" : gt3x_path,
        "ggir_raw" : ggir_path
    }
    with open(filepath, 'w') as f:
        json.dump(new_config, f)


def run(args:str =None) -> None:  # noqa: D103
    parser = ArgumentParser( description= "This is wristpy, a work in progress. At the \
                            moment you have to put in both a raw file, and a ggir \
                            output file for comparison.Please double check that \
                            default file paths are appropriately configured.")
    
    parser.add_argument("-f","--gt3xfile", help= "file name for gt3x file. Make sure \
                        file path is properly configured in config.json file. Modify \
                        path with --config or -c as needed", type = str)
    
    parser.add_argument("-g","--ggirfile", help= "file name for ggir output file. Make\
                        sure that file path is properly configured in config.json file.\
                        Modify path with --config or -c as needed.", type = str)
    
    parser.add_argument("-s", "--start", type = str, help= "The first date you want data\
                         from in the format of YYYY-MM-DD HH:MM:SS leave empty to select all dates.")  # noqa: E501
    
    parser.add_argument("-e", "--end", type = str, help = "The last date you want data \
                         from in the format of YYYY-MM-DD HH:MM:SS leave empty to select all dates.")  # noqa: E501
    
    parser.add_argument("-m", "--measures", choices = ['ENMO', 'anglez', 'qq', 'ba'], nargs= "+",  # noqa: E501
                        help = "Select which measures you would liketo plot. Options \
                        include ENMO, anglez, qq, ba")
    
    parser.add_argument('-c', '--config', type = str, help = " Change file paths for \
                        gt3x raw file and ggir output file, in that order.")
    
    arguments = parser.parse_args(args) if args else parser.parse_args()

    path_dict = load_config('config.json')

    if arguments.config:
        new_gt3x_path, new_ggir_path = arguments.config
        update_config(
            'config.json', 
            gt3x_path= new_gt3x_path, 
            ggir_path= new_ggir_path)
        path_dict = load_config('config.json')

    
    gt3x_raw_path = path_dict['gt3x_raw'] + arguments.gt3xfile
    ggir_output_path = path_dict['ggir_output'] + arguments.ggirfile

    test_config = wristpy.common.data_model.Config(gt3x_raw_path, gt3x_raw_path)  # noqa: E501
    test_data = gt3x.load(test_config.path_input)
    test_output = calibration.start_ggir_calibration(test_data)

    metrics_calc.calc_base_metrics(test_output)

    metrics_calc.calc_epoch1_metrics(test_output)

    ggir_data = compare_dataframes.load_ggir_output(ggir_output_path)

    

    difference_df, outputdata_trimmed = compare_dataframes.compare(
                                    ggir_dataframe= ggir_data,
                                    wristpy_dataframe=test_output)
    
    #If subset of dates given, select data for those dates only.
    if arguments.start or arguments.end:
        diff_df_slice, ggir_df_slice, wristpy_df_slice= compare_dataframes.select_dates(
            difference_df=difference_df, 
            outputdata_trimmed=outputdata_trimmed,
            ggir_data=ggir_data,
            start=arguments.start,
            end=arguments.end
            )
        
        for measure in arguments.measures:
            compare_dataframes.plot_diff(
                outputdata_trimmed = wristpy_df_slice, 
                ggir_dataframe = ggir_df_slice, 
                difference_df = diff_df_slice,
                measures=measure, 
                opacity= 0.5)
    #If not use the entire dataframes.            
    else:
        for measure in arguments.measures:
            compare_dataframes.plot_diff(
                outputdata_trimmed = outputdata_trimmed, 
                ggir_dataframe = ggir_data, 
                difference_df = difference_df,
                measures=measure, 
                opacity= 0.5)

if __name__ == "__main__":
    run()