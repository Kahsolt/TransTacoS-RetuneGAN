ifeq ($(shell uname -s), Linux)
	BASE_PATH=~/Data
else
	BASE_PATH=D:/Desktop/Workspace/Data
endif

DATASET=DataBaker
DATA_PATH=$(BASE_PATH)/$(DATASET).tts_processed
#LOG_PATH=$(BASE_PATH)/rtg-$(DATASET).$(VER)
LOG_PATH=$(BASE_PATH)/rtg-$(DATASET)

.PHONY: train test server clean stat

train:
	python train.py \
		--data_dp $(DATA_PATH) \
		--log_path $(LOG_PATH) \
		--epochs 3100

finetune:
	python train.py \
		--finetune \
		--data_dp $(DATA_PATH) \
		--log_path $(LOG_PATH) \
		--epochs 3100

test:
	python infer.py \
	  --log_path $(LOG_PATH) \
		--input_path test

server:
	python server.py \
	  --log_path $(LOG_PATH)

stat:
	tensorboard \
	  --logdir $(LOG_PATH) \
		--port 5101

clean:
	rm -rf $(LOG_PATH)
