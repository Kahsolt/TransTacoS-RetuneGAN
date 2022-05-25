ifeq ($(shell uname -s), Linux)
	BASE_PATH=~/Data
else
	BASE_PATH=D:/Desktop/Workspace/Data
endif

DATASET=DataBaker
#LOG_NAME=tts-$(DATASET).$(VER)
LOG_NAME=tts-$(DATASET)
LOG_PATH=${BASE_PATH}/$(LOG_NAME)


.PHONY: train test server clean stat

train:
	python train.py \
	  --base_dir $(BASE_PATH) \
	  --input $(DATASET).tts_processed/train.txt \
	  --name $(LOG_NAME) \
	  --summary_interval 500 \
	  --checkpoint_interval 1000

preprocess:
	python preprocess.py \
	  --base_dir $(BASE_PATH) \
	  --out_dir $(DATASET).tts_processed \
	  --dataset $(shell echo $(DATASET) | tr '[A-Z]' '[a-z]')

server:
	python server.py \
	  --log_path $(LOG_PATH)

test_server:
	python server.py \
	  --log_path $(LOG_PATH) \
		--port 5103

stat:
	tensorboard \
	  --logdir $(LOG_PATH) \
	  --port 5103

clean:
	rm -rf $(LOG_PATH)
