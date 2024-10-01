# EMBEDDRIX

Embeddrix is a microservice for generating text embeddings using jina-embeddings-v3.

## Prequisites

- NVIDIA GPU with CUDA and at least 2.5 GiB of VRAM
- Docker and Docker compose (optional)
- NVIDIA Container Toolkit (for docker to have access to GPUs)

## Deploy

### Manual

Run an instance of memcached. Define `MEMCACHED_HOST` and `MEMCACHED_PORT` env variables.
Install python 3.12 (might work on earlier versions, haven't tested) and poetry.

Run the following:

```sh
poetry install --no-root --no-interaction --no-ansi
poetry run pip install --no-cache-dir --use-pep517 --no-build-isolation "flash-attn (==2.6.3)" # Becuase flash-attn wouldn't install otherwise for some reason
```

Then you can run the app using:

```sh
poetry run uvicorn embeddrix.app:app --host 0.0.0.0 --port 44777
```

Alter host and port values to your liking.

### Via docker compose 

If you can't be bothered to configure the env to run this crap, clone the repo and run:

```sh
docker compose up
```

Docker compose file assumes you got an nvidia gpu along with nvidia container toolkit,
so you gotta install it. Performance on CPU is suboptimal. If you wanna AMD support
figure it out yourself, I ain't got one. PRs welcome though.

Also you can use dockerfile in your own compose apps, obvsly... use `docker-compose.yml` as reference.
