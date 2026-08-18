.DEFAULT_GOAL := help
COMPOSE := docker compose

help:  ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

up:  ## Build, start Postgres, run the pipeline, serve the API on :8000
	$(COMPOSE) up --build -d
	@echo "API: http://localhost:8000/docs"

down:  ## Stop everything (keeps the database volume)
	$(COMPOSE) down

clean:  ## Stop everything and delete the database volume
	$(COMPOSE) down -v

logs:  ## Follow service logs
	$(COMPOSE) logs -f

pipeline:  ## Re-run migrate -> seed -> chunk -> embed
	$(COMPOSE) run --rm pipeline

shell:  ## psql into the database
	$(COMPOSE) exec postgres psql -U drr -d drr_rag

test:  ## Run the test suite
	$(COMPOSE) run --rm --entrypoint pytest pipeline -q

eval:  ## Run the retrieval evaluation and write eval/results/
	$(COMPOSE) run --rm pipeline python -m eval.run_eval

eval-weights:  ## Evaluate and sweep the RRF arm weights
	$(COMPOSE) run --rm pipeline python -m eval.run_eval --weights

eval-sweep:  ## Re-chunk at 0/25/50% overlap and evaluate each
	$(COMPOSE) run --rm pipeline python -m eval.run_eval --sweep

corpus:  ## Re-fetch the frozen corpus from the public APIs
	python scripts/fetch_corpus.py

.PHONY: help up down clean logs pipeline shell test eval eval-weights eval-sweep corpus
