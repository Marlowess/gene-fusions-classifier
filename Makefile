SCRIPT_INTERPETER = python3

SCRIPT_2_TEST = test_feature.py
TESTS_DIR = tests

SCRIPT_ANALYSIS = main_2.py
NETWORK_NAME = EmbeddingLstm
ARGS_ANALYSIS = --validation --load_network $(NETWORK_NAME)

# ---------------------------- #
# Test Seutp Project's subidrs #
# ---------------------------- #
SCRIPT_TEST_ENVIRONMENT_SETUP = script_environment_setup_test.py
ARGS_TEST_ENVIRONMENT_SETUP = --validation --train

# ---------------------------- #
# Test Load Project's data     #
# ---------------------------- #
SCRIPT_TEST_FETCH_ANDPREPROCESS = script_fetch_and_preprocess_test.py
ARGS_TEST_FETCH_ANDPREPROCESS = --validation --train

# ---------------------------- #
# Test Pipeline for Analyses   #
# ---------------------------- #
SCRIPT_TEST_PIPELINE = script_pipeline_test.py
ARGS_TEST_FETCH_ANDPREPROCESS = --validation --load_network EmbeddingLstm

# ---------------------------- ---------------------#
# Test Analys model embeddign bidirectional protein #
# ------------------------------------------------- #
PROGRAM_ENTRY_POINT_M1 = run_analysis.py
ARGS_VALIDATION_M1 = --validation --load_network ModelEmbeddingBidirectProtein --sequence_type protein --lr 5e-4 
ARGS_VALIDATION_TRAIN_M1 = 
ARGS_VALIDATION_TEST_M1 = 

run_analysis:
	cp $(TESTS_DIR)/$(SCRIPT_TEST_PIPELINE) $(SCRIPT_ANALYSIS)
	$(SCRIPT_INTERPETER) $(SCRIPT_ANALYSIS) $(ARGS_ANALYSIS)

run_validation_on_model_embedding_bidirectional_protein:
	$(SCRIPT_INTERPETER) $(PROGRAM_ENTRY_POINT_M1) $(ARGS_VALIDATION_M1)

test_setup_environment_for_analysis:
	cp $(TESTS_DIR)/$(SCRIPT_TEST_ENVIRONMENT_SETUP) $(SCRIPT_2_TEST)
	$(SCRIPT_INTERPETER) $(SCRIPT_2_TEST) $(ARGS_TEST_ENVIRONMENT_SETUP)
	rm -f $(SCRIPT_2_TEST)

test_fetch_data_and_preprocess_for_analysis:
	cp $(TESTS_DIR)/$(SCRIPT_TEST_FETCH_ANDPREPROCESS) $(SCRIPT_2_TEST)
	$(SCRIPT_INTERPETER) $(SCRIPT_2_TEST) $(ARGS_TEST_FETCH_ANDPREPROCESS)
	rm -f $(SCRIPT_2_TEST)

test_pipeline_for_analysis:
	cp $(TESTS_DIR)/$(SCRIPT_TEST_PIPELINE) $(SCRIPT_2_TEST)
	$(SCRIPT_INTERPETER) $(SCRIPT_2_TEST) $(ARGS_TEST_FETCH_ANDPREPROCESS)
	rm -f $(SCRIPT_2_TEST)

install_libraries_for_graphviz:
	pip install pydot
	pip install pydotplus
	sudo apt-get install graphviz
