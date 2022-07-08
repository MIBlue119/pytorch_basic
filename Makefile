# Ref:https://github.com/coqui-ai/TTS/blob/dev/Makefile
.DEFAULT_GOAL := help 
.PHONY: help install 

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install the basic requirements
	pip install -r requirements.txt