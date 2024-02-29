from argparse import ArgumentParser

import wristpy
from wristpy.common import data_model
from wristpy.ggir import calibration, compare_dataframes, metrics_calc
from wristpy.io.loaders import gt3x


def run(args:str =None) -> None:  # noqa: D103
    parser = ArgumentParser( description= """This is wristpy! It's not GGIR and that
                             alone is good news!""")
    parser.add_argument("-f", "--gt3xpath", help= "The file path to the gt3x file",
                          type = str)
    parser.add_argument("-g", "--ggirpath", help= "Path to ggir output",
                        type = str)
    parser.add_argument("-i", "--indices", help="""The selection of your data you want to process and plot. 
                        in the form of a range. Default None for all data """)  # noqa: E501
    parser.add_argument("-m", "--measures", choices = ['ENMO', 'anglez', 'qq', 'ba'], nargs= "+",
                        help = """Select which measures you would liketo plot. Options 
                        include ENMO, anglez, qq, ba""")
    arguments = parser.parse_args(args) if args else parser.parse_args()

    for measure in arguments.measures:
        print(measure)

    test_config = wristpy.common.data_model.Config(arguments.gt3xpath, arguments.gt3xpath)  # noqa: E501
    test_data = gt3x.load(test_config.path_input)
    test_output = calibration.start_ggir_calibration(test_data)

    metrics_calc.calc_base_metrics(test_output)

    metrics_calc.calc_epoch1_metrics(test_output)

    ggir_data = compare_dataframes.load_ggir_output(arguments.ggirpath)
    difference_df, outputdata_trimmed = compare_dataframes.compare(
                                    ggir_dataframe= ggir_data,
                                    wristpy_dataframe=test_output)
    

    for measure in arguments.measures:
        compare_dataframes.plot_diff(
            outputdata_trimmed = outputdata_trimmed, 
            ggir_dataframe = ggir_data, 
            difference_df = difference_df,
            measures=measure, 
            opacity= 0.5)

if __name__ == "__main__":
    run()