INTERPETER = python3

# ------------------------------ #
# SRC FILES                      #
# ------------------------------ #

SCRIPT_NAME = src_script.py
MAIN_SCRIPT = main.py
PROGRAM_NAME = generate_hyperparams_combinations

# ------------------------------ #
# COMMAND ARSG                   #
# ------------------------------ #
INPUT_FILE = --input_file ./hyper_parameters_configurations/hyper_params_lstm_1_layer.json
OUTPUT_FILE = --output_file ./tables_csv/hyperparams_combinations.csv

ARGS = $(INPUT_FILE) $(OUTPUT_FILE)

run_time_series_forecasting: clear_and_setup
	$(INTERPETER) $(PROGRAM_NAME) $(ARGS)

run_help_tool: clear_and_setup
	$(INTERPETER) $(PROGRAM_NAME) -h

clear_and_setup:
	clear
	ln -sfn $(SCRIPT_NAME) $(MAIN_SCRIPT)
	ln -sfn $(MAIN_SCRIPT) $(PROGRAM_NAME)
	chmod u+x $(PROGRAM_NAME)