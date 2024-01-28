<p align="center">
  <img alt="Latte logo" src="https://github.com/mcobzarenco/latte/assets/797170/fdff8b28-7b00-47ae-bfd0-05efcc018866" width="400px">
</p>

# Latte: Latent Attention for Linear Time Transformers

This repository is an implementaion of latent attention (latte) for language modelling.

## Get started

Get started by cloning the repository.

We use [PDM](https://pdm.fming.dev/latest/) to manage Python packages and dependencies and use
Python 3.10 (consider [pyenv](https://github.com/pyenv/pyenv) as a simple Python version management
solution). `pdm` is a modern alternative to `pip` / `poetry`. `pdm` manages a virtualenv
locally in the project directory itself (in a directory `.venv`).

It keeps dependencies from `pyproject.toml` in sync with the virtual environment. This makes it easy
to have a python environment specific to the project and ensure we all run the same dependencies.
Similar to how `npm`, `yarn` for JS/TS or `cargo` for Rust work.

### Installation

Once you have `pdm` and the right python version installed, run

```
pdm install
```

## Scripts

 - `pdm run train` will train a model 
 - `pdm run preprocess-tiny-stories` will download and preprocess the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset locally

