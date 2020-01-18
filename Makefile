SCRIPT_INTERPETER = python3

BASE_DIR_RESULTS = bioinfo_project

SCRIPT_2_TEST = test_feature.py
TESTS_DIR = tests

SCRIPT_ANALYSIS = main_2.py
NETWORK_NAME = ModelEmbeddingBidirect
ARGS_ANALYSIS = --validation --load_network $(NETWORK_NAME)

# ---------------------------- #
# Test Seutp Project's subidrs #
# ---------------------------- #
SCRIPT_TEST_ENVIRONMENT_SETUP = script_environment_setup_test.py
ARGS_TEST_ENVIRONMENT_SETUP = --validation --train --network_parameters models/ModelEmbeddingBidirect.json --load_network ModelEmbeddingBidirect

# ---------------------------- #
# Test Load Project's data     #
# ---------------------------- #
SCRIPT_TEST_FETCH_AND_PRE_PROCESS = script_fetch_and_preprocess_test.py
ARGS_TEST_FETCH_AND_PREPROCESS = --validation --train --network_parameters models/ModelEmbeddingBidirect.json --load_network ModelEmbeddingBidirect

# ---------------------------- #
# Test Pipeline for Analyses   #
# ---------------------------- #
SCRIPT_TEST_PIPELINE = script_pipeline_test.py
ARGS_TEST_PIPELINE = --validation --network_parameters models/ModelEmbeddingUnidirect.json --load_network ModelEmbeddingUnidirect --sequence_type protein

# ---------------------------- ---------------------#
# Test Analys model embeddign bidirectional protein #
# ------------------------------------------------- #
PROGRAM_ENTRY_POINT_M1 = run_analysis.py
ARGS_VALIDATION_M1 = --validation --network_parameters models/ModelEmbeddingBidirect.json --load_network ModelEmbeddingBidirect --sequence_type protein --lr 5e-4 
ARGS_VALIDATION_TRAIN_M1 = 
ARGS_VALIDATION_TEST_M1 = 

# ---------------------------- ---------------------#
# Test Analys model one hot encoding protein        #
# ------------------------------------------------- #
PROGRAM_ENTRY_POINT_M2 = run_analysis.py
ARGS_VALIDATION_M2 = --validation --load_network OneHotEncodedLstm --sequence_type protein 
ARGS_VALIDATION_TRAIN_M2 = --validation --train --load_network OneHotEncodedLstm --sequence_type protein 
ARGS_TRAIN_M2 = --train --load_network OneHotEncodedLstm --sequence_type protein steps 200
ARGS_VALIDATION_TRAIN_TEST_M2 = --validation --train --test --load_network OneHotEncodedLstm --sequence_type protein
ARGS_TRAIN_TEST_M2 = --train --test --load_network OneHotEncodedLstm --sequence_type protein --steps 10

run_analysis: setup_before_run_task
	cp $(TESTS_DIR)/$(SCRIPT_TEST_PIPELINE) $(SCRIPT_ANALYSIS)
	$(SCRIPT_INTERPETER) $(SCRIPT_ANALYSIS) $(ARGS_ANALYSIS)

run_validation_on_model_embedding_bidirectional_protein:
	$(SCRIPT_INTERPETER) $(PROGRAM_ENTRY_POINT_M1) $(ARGS_VALIDATION_M1)

run_validation_on_model_one_hot_encoding_protein:
	$(SCRIPT_INTERPETER) $(PROGRAM_ENTRY_POINT_M2) $(ARGS_VALIDATION_M2)

run_validation_train_on_model_one_hot_encoding_protein:
	$(SCRIPT_INTERPETER) $(PROGRAM_ENTRY_POINT_M2) $(ARGS_VALIDATION_TRAIN_M2)

run_validation_train_test_on_model_one_hot_encoding_protein:
	$(SCRIPT_INTERPETER) $(PROGRAM_ENTRY_POINT_M2) $(ARGS_VALIDATION_TRAIN_TEST_M2)

run_train_test_on_model_one_hot_encoding_protein:
	$(SCRIPT_INTERPETER) $(PROGRAM_ENTRY_POINT_M2) $(ARGS_TRAIN_TEST_M2)

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


# ---------------------------- ---------------------#
# MANAGEMENT - SECTION                              #
# ------------------------------------------------- #
setup_before_run_task:
	clear

install_libraries_for_graphviz:
	pip install pydot
	pip install pydotplus
	sudo apt-get install graphviz

# Clear directory with subdirectories corresponind to
# different runs with their results
clear_result_dirs: setup_before_run_task
	rm -fr $(BASE_DIR_RESULTS)


