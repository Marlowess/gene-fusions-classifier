# =============================================================================================== #
#                                       SRC: Makefile                                             #
# =============================================================================================== #

# =============================================================================================== #
# VARIABLES SECTION                                                                               #
# =============================================================================================== #

REQUIREMNTES_FILE  = requirements.txt
LOCAL_PIP = pip3
CONDA_PIP = '...'

PROJECT_CODEBASE = ./
ARCHIVE_PROJECT_NAME_COLAB = project-genes-fusions-classifier.zip
EXCLUDED_FILES = \
	'*.vscode/*' \
	'*.git/*' \
	.gitignore \
	'*__pycache__/*' \
	./README.md \
	./test_feature.py \
	'*notebooks/*' \
	'*tmp/*' \
	'*data/*' \
	'*tests/*' \
	'*scripts/*' \
	*.png

# Here - Specify which python interpreter to employ.
SCRIPT_INTERPETER = python3 

# Here - Specify which is the root directory where all results
# from different analyses will be stored, for later investingations 
# about  results, logs and image files.
BASE_DIR_RESULTS = bioinfo_project 

# Here - There are some variable exploited in order
# to run some local tests.
#
# See tasks related to test for more details.
SCRIPT_2_TEST = test_feature.py
TESTS_DIR = tests

# Here - There are two variable used to run the
# classifier tool for genes fusions recognizing:
# - SCRIPT_ANALYSIS, standas for the `main.py` file which is used to
#   write the program;
# - PROGRAM_ANALYSIS, is a symbolic link that can be used to run
#   the program written above without taking care about how it was implemented
#   as weel as without knowing which kind of programming language has been adopted.
#
# See task named `run_analysis` for details how both them have been employed.
SCRIPT_ANALYSIS = main.py
PROGRAM_ANALYSIS = genes_fusions_classifier

# =============================================================================================== #
# VARIABLES DEFINED TO RUN TESTS WITHIN PROPER TASKS                                              #
# =============================================================================================== #

# ---------------------------- #
# A Test                       #
# ---------------------------- #
NETWORK_NAME = ModelEmbeddingBidirect
ARGS_ANALYSIS = --validation --load_network $(NETWORK_NAME)

# ---------------------------- #
# Test Setup Project's subidrs #
# ---------------------------- #
SCRIPT_TEST_ENVIRONMENT_SETUP = script_environment_setup_test.py
ARGS_TEST_ENVIRONMENT_SETUP = --validation --train --network_parameters models/ModelEmbeddingBidirect.json --load_network ModelEmbeddingBidirect

# ---------------------------- #
# Test Load Project's data     #
# ---------------------------- #
SCRIPT_TEST_FETCH_AND_PRE_PROCESS = script_fetch_and_preprocess_test.py
ARGS_TEST_FETCH_AND_PREPROCESS = --validation --train --sequence_type dna --network_parameters models/ModelEmbeddingBidirect.json --load_network ModelEmbeddingBidirect

# ---------------------------- #
# Test Pipeline for Analyses   #
# ---------------------------- #
SCRIPT_TEST_PIPELINE = script_pipeline_test.py
ARGS_TEST_PIPELINE = --validation --network_parameters models/ModelEmbeddingBidirect.json --load_network ModelEmbeddingBidirect --sequence_type protein

# ---------------------------- #
# Test Spredsheet for Analyses #
# ---------------------------- #
SCRIPT_TEST_SPREDSHEET = script_create_spredsheet.py
ARGS_TEST_SPREDSHEET = --validation --network_parameters models/ModelEmbeddingBidirect.json --load_network ModelEmbeddingBidirect --sequence_type protein

# ---------------------------- ---------------------#
# Test Analys model embeddign bidirectional protein #
# ------------------------------------------------- #
PROGRAM_ENTRY_POINT_M1 = main.py
ARGS_VALIDATION_M1 = --validation --network_parameters models/ModelEmbeddingBidirect.json --load_network ModelEmbeddingBidirect --sequence_type protein --lr 5e-4 
ARGS_VALIDATION_TRAIN_M1 = 
ARGS_VALIDATION_TEST_M1 = 

# ---------------------------- ---------------------#
# Test Analys model one hot encoding protein        #
# ------------------------------------------------- #
PROGRAM_ENTRY_POINT_M2 = run_analysis.py
ARGS_VALIDATION_M2 = --validation --network_parameters models/ModelOneHotProtein.json --load_network ModelOneHotProtein --sequence_type protein --onehot_flag
ARGS_VALIDATION_M3 = --validation --network_parameters models/ModelConvBidirect.json --load_network ModelConvBidirect --sequence_type protein --onehot_flag
ARGS_VALIDATION_TRAIN_M2 = --validation --train --network_parameters models/ModelOneHotProtein.json --load_network ModelOneHotProtein --sequence_type protein --onehot_flag
ARGS_TRAIN_M2 = --train --load_network --network_parameters models/ModelOneHotProtein.json --load_network ModelOneHotProtein --sequence_type protein steps 200 --onehot_flag
ARGS_VALIDATION_TRAIN_TEST_M2 = --validation --train --test --load_network ModelOneHotProtein --network_parameters models/ModelOneHotProtein.json --sequence_type protein --onehot_flag
ARGS_TRAIN_TEST_M2 = --train --test --load_network ModelOneHotProtein --sequence_type protein --network_parameters models/ModelOneHotProtein.json --early_stopping_epoch 10 --onehot_flag


# ---------------------------- ------------------------------ #
# Test Analys model one hot encoding dna (Frank-Added)        #
# ----------------------------------------------------------- #
ULSTM = genes_fusions_classifier
ARGS_TRAIN_ULSTM = --train \
	--load_network WrappedRawModel \
	--sequence_type dna \
	--network_parameters models/experimental_simple_models/model_dna_embedding_unidirect.json \
	--num_epochs 2 \
	--batch_size 32 \
# =============================================================================================== #
# TASKS SECTION                                                                                   #
# =============================================================================================== #

# ---------------------------- ---------------------#
# RUN ANALYSES - SECTION                            #
# ------------------------------------------------- #

run_help_classifier_tool: setup_before_run_task
	# cp $(TESTS_DIR)/$(SCRIPT_TEST_PIPELINE) $(SCRIPT_ANALYSIS)
	ln -sfn $(SCRIPT_ANALYSIS) $(PROGRAM_ANALYSIS)
	chmod u+x $(PROGRAM_ANALYSIS)
	$(SCRIPT_INTERPETER) $(PROGRAM_ANALYSIS) -h

run_analysis: setup_before_run_task
	# cp $(TESTS_DIR)/$(SCRIPT_TEST_PIPELINE) $(SCRIPT_ANALYSIS)
	cp -sfn $(SCRIPT_ANALYSIS) $(PROGRAM_ANALYSIS)
	chmod u+x $(PROGRAM_ANALYSIS)
	$(SCRIPT_INTERPETER) $(SCRIPT_ANALYSIS) $(ARGS_ANALYSIS)

run_validation_on_model_embedding_bidirectional_protein:
	$(SCRIPT_INTERPETER) $(PROGRAM_ENTRY_POINT_M1) $(ARGS_VALIDATION_M1)

run_validation_on_model_conv_bidirect:
	$(SCRIPT_INTERPETER) $(PROGRAM_ENTRY_POINT_M1) $(ARGS_VALIDATION_M3)

run_validation_on_model_one_hot_encoding_protein:
	$(SCRIPT_INTERPETER) $(PROGRAM_ENTRY_POINT_M2) $(ARGS_VALIDATION_M2)

run_validation_train_on_model_one_hot_encoding_protein:
	$(SCRIPT_INTERPETER) $(PROGRAM_ENTRY_POINT_M2) $(ARGS_VALIDATION_TRAIN_M2)

run_validation_train_test_on_model_one_hot_encoding_protein:
	$(SCRIPT_INTERPETER) $(PROGRAM_ENTRY_POINT_M2) $(ARGS_VALIDATION_TRAIN_TEST_M2)

run_train_test_on_model_one_hot_encoding_protein:
	$(SCRIPT_INTERPETER) $(PROGRAM_ENTRY_POINT_M2) $(ARGS_TRAIN_TEST_M2)

run_train_ULSTM:
	$(SCRIPT_INTERPETER) $(ULSTM) $(ARGS_TRAIN_ULSTM)

# ---------------------------- ---------------------#
# TESTS - SECTION                                   #
# ------------------------------------------------- #
test_setup_environment_for_analysis: setup_before_run_task
	cp $(TESTS_DIR)/$(SCRIPT_TEST_ENVIRONMENT_SETUP) $(SCRIPT_2_TEST)
	$(SCRIPT_INTERPETER) $(SCRIPT_2_TEST) $(ARGS_TEST_ENVIRONMENT_SETUP)
	rm -f $(SCRIPT_2_TEST)

test_fetch_data_and_preprocess_for_analysis: setup_before_run_task
	cp $(TESTS_DIR)/$(SCRIPT_TEST_FETCH_AND_PRE_PROCESS) $(SCRIPT_2_TEST)
	$(SCRIPT_INTERPETER) $(SCRIPT_2_TEST) $(ARGS_TEST_FETCH_AND_PREPROCESS)
	rm -f $(SCRIPT_2_TEST)

test_pipeline_for_analysis: setup_before_run_task
	cp $(TESTS_DIR)/$(SCRIPT_TEST_PIPELINE) $(SCRIPT_2_TEST)
	$(SCRIPT_INTERPETER) $(SCRIPT_2_TEST) $(ARGS_TEST_PIPELINE)
	rm -f $(SCRIPT_2_TEST)

test_spredsheet_creation_for_analysis: setup_before_run_task
	cp $(TESTS_DIR)/$(SCRIPT_TEST_SPREDSHEET) $(SCRIPT_2_TEST)
	$(SCRIPT_INTERPETER) $(SCRIPT_2_TEST) $(ARGS_TEST_SPREDSHEET)
	rm -f $(SCRIPT_2_TEST)

# ---------------------------- ---------------------#
# MANAGEMENT - SECTION                              #
# ------------------------------------------------- #
setup_before_run_task:
	clear

install_requirements_via_pip:
	$(LOCAL_PIP) install -r $(REQUIREMNTES_FILE)
install_requirements_via_conda:	
	$(CONDA_PIP) install -r $(REQUIREMNTES_FILE)

install_libraries_for_graphviz:
	pip install pydot
	pip install pydotplus
	sudo apt-get install graphviz

# Clear directory with subdirectories corresponind to
# different runs with their results.
clear_result_dirs: setup_before_run_task
	rm -fr $(BASE_DIR_RESULTS)

# different runs with their results.
clear_result_dirs_from_tests: setup_before_run_task
	bash ./scripts/script_clear_results_dir_from_tests.sh ./bioinfo_project --not-cancel

build_zip_to_run_on_colab: clear_result_dirs
	rm -f ../$(ARCHIVE_PROJECT_NAME_COLAB)
	zip -r  ../$(ARCHIVE_PROJECT_NAME_COLAB) $(PROJECT_CODEBASE) -x $(EXCLUDED_FILES)



