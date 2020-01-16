SCRIPT_INTERPETER = python3

SCRIPT_2_TEST = test_feature.py
TESTS_DIR = tests

# ---------------------------- #
# Test Seutp Project's subidrs #
# ---------------------------- #
SCRIPT_TEST_ENVIRONMENT_SETUP = script_environment_setup_test.py
ARGS_TEST_ENVIRONMENT_SETUP = 

# ---------------------------- #
# Test Load Project's data     #
# ---------------------------- #
SCRIPT_TEST_FETCH_ANDPREPROCESS = script_fetch_and_preprocess_test.py
ARGS_TEST_FETCH_ANDPREPROCESS = 

test_setup_environment_for_analysis:
	cp $(TESTS_DIR)/$(SCRIPT_TEST_ENVIRONMENT_SETUP) $(SCRIPT_2_TEST)
	$(SCRIPT_INTERPETER) $(SCRIPT_2_TEST) $(ARGS_TEST_ENVIRONMENT_SETUP)
	rm -f $(SCRIPT_2_TEST)

test_fetch_data_and_preprocess_for_analysis:
	cp $(TESTS_DIR)/$(SCRIPT_TEST_FETCH_ANDPREPROCESS) $(SCRIPT_2_TEST)
	$(SCRIPT_INTERPETER) $(SCRIPT_2_TEST) $(ARGS_TEST_FETCH_ANDPREPROCESS)
	rm -f $(SCRIPT_2_TEST)