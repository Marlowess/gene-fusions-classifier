
EXAMPLE_GRU_IMAGE_MODEL = example_gru_model_sequential.png
EXAMPLE_LSTM_IMAGE_MODEL = example_lstm_model_sequential.png

run_main_script: clear_and_setup
	chmod u+x main.py
	python3 main.py
	xdg-open $(EXAMPLE_GRU_IMAGE_MODEL)
	xdg-open $(EXAMPLE_LSTM_IMAGE_MODEL)

run_gru_sequential_example:
	clear
	chmod u+x main.py
	python3 main.py
	xdg-open $(EXAMPLE_GRU_IMAGE_MODEL)

run_lstm_sequential_example:
	clear
	chmod u+x main.py
	python3 main.py
	xdg-open $(EXAMPLE_GRU_IMAGE_MODEL)

clear_and_setup:
	clear
	rm -fr *.png