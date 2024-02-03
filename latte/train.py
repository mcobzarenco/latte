import dataclasses
import multiprocessing
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import flax
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
import tiktoken
import wandb
from flax.core.frozen_dict import FrozenDict, freeze
from jax import Array
from numpy.typing import NDArray
from optax._src.base import GradientTransformation
from simple_parsing import parse as parse_args
from simple_parsing.wrappers.field_wrapper import DashVariant
from torch.utils.data import DataLoader, Dataset

from .model import Transformer

GPT2_TOKENIZER = tiktoken.get_encoding("gpt2")

# JAX_DISABLE_JIT=True


@dataclass
class Arguments:
    """
    A tool for creating tokenizers from datasets
    """

    # # The root directory where all datasets will be stored
    # datasets_dir: Path = DATASETS_ROOT_PATH

    # # Path to an output directory where trained tokenizers will be stored
    # output_dir: Path = Path(".")

    # # Which type of tokenizer to train
    # kind: TokenizerKind = TokenizerKind.WORD_PIECE

    # Number of layers in the transformer
    num_layers: int = 2

    # Dtype
    dtype: str = "float32"

    # Number of heads in multihead attention
    num_heads: int = 4

    # # Checkpoint
    # checkpoint_dir: Path =

    # Embedding size
    embedding_size: int = 256

    # Context length in number of tokens
    context_size: int = 1024

    # Sample length in number of tokens
    num_samples: int = 4

    # Sample length in number of tokens
    sample_size: int = 64

    # Batch size
    batch_size: int = 16

    # Learning rate
    learning_rate: float = 1e-3

    # Weight decay
    weight_decay: float = 0.01

    # Number of training iterations
    num_iters: int = 10000

    # Number of training iterations
    num_warmup_iters: int = 100

    # Number of workers for torch.data.DataLoader
    num_loader_workers: int = 4

    # Log to wandb
    wandb: bool = False

    # Number of training iterations
    wandb_project: str = "latte"

    # Name of wandb run
    wandb_run_name: str | None = None

    # Number of training iterations
    attention_type: Literal["latte", "latte-wrap", "standard"] = "latte"

    # Gradient clipping value
    gradient_clip: float = 1.0

    # Whether to show debug logs
    verbose: bool = False


def sample(
    apply_fn: Callable,
    weights: dict,
    num_samples: int,
    sample_size: int,
    key: Array,
) -> Array:
    def sample_next_token(
        token_index: int,
        indices_and_key: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        (indices, key) = indices_and_key

        (logits, _) = apply_fn(weights, indices=indices, deterministic=True)

        key, key_generation = jax.random.split(key)
        next_indices = jax.random.categorical(
            key_generation, logits[:, token_index - 1, :], axis=-1
        ).astype(jnp.uint16)

        indices = indices.at[:, token_index].set(next_indices)
        return (indices, key)

    sample = jnp.full((num_samples, sample_size), fill_value=GPT2_TOKENIZER.eot_token)
    (indices, _) = jax.lax.fori_loop(
        lower=1,
        upper=sample_size,
        body_fun=sample_next_token,
        init_val=(sample, key),
    )
    return indices


class TrainState(flax.struct.PyTreeNode):
    step: int
    gradient_norm: Array
    apply_fn: Callable = flax.struct.field(pytree_node=False)
    optimizer: GradientTransformation = flax.struct.field(pytree_node=False)
    weights: FrozenDict[str, Any]
    optimizer_state: optax.OptState

    def apply_gradients(self, *, grads: Array, **kwargs):
        updates, new_optimizer_state = self.optimizer.update(
            grads,
            self.optimizer_state,
            self.weights,
        )
        new_weights = optax.apply_updates(self.weights, updates)
        gradient_norm = optax.global_norm(updates)

        return self.replace(
            step=self.step + 1,
            gradient_norm=gradient_norm,
            weights=new_weights,
            optimizer_state=new_optimizer_state,
            **kwargs,
        )

    @classmethod
    def create(
        cls,
        *,
        apply_fn: Callable,
        weights: FrozenDict[str, Any],
        optimizer: GradientTransformation,
    ):
        optimizer_state = optimizer.init(weights)
        return cls(
            step=0,
            gradient_norm=jnp.array(0.0),
            apply_fn=apply_fn,
            weights=weights,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
        )


def train_step(train_state: TrainState, inputs: Array, targets: Array) -> tuple[TrainState, Array]:
    def loss_fn(weights: FrozenDict[str, Any]):
        (_logits, loss) = train_state.apply_fn(
            weights,
            indices=inputs,
            targets=targets,
            deterministic=False,
        )
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(train_state.weights)
    new_train_state = train_state.apply_gradients(grads=grads)

    return new_train_state, loss


def sample_batch(
    *,
    indices: NDArray[np.uint16],
    batch_size: int,
    context_size: int,
    key: Array,
) -> tuple[Array, Array]:
    batch_indices = jax.random.randint(
        key,
        minval=0,
        maxval=len(indices) - context_size - 1,
        shape=(batch_size,),
    )
    x = jnp.stack([indices[index : index + context_size] for index in batch_indices])
    y = jnp.stack([indices[index + 1 : index + context_size + 1] for index in batch_indices])

    return (x, y)


@dataclass(frozen=True)
class Datapoint:
    # S = sequence length
    inputs: NDArray[np.uint16]  # (S,)
    targets: NDArray[np.uint16]  # (S,)


@dataclass(frozen=True)
class DatapointBatch:
    # N = batch size
    # S = sequence length
    inputs: NDArray[np.uint16]  # (N, S)
    targets: NDArray[np.uint16]  # (N, S)


class WindowDataset(Dataset[Datapoint]):
    def __init__(self, tokens: NDArray[np.uint16], window_size: int):
        assert len(tokens.shape) == 1 and tokens.dtype == np.uint16, (
            f"Invalid shape {tokens.shape}",
        )
        assert window_size > 1, f"Invalid window_size: {window_size}"

        self._tokens = tokens
        self._window_size = window_size

    def __len__(self) -> int:
        return len(self._tokens) - self._window_size - 1

    def __getitem__(self, window_index: int) -> Datapoint:
        inputs = self._tokens[window_index : window_index + self._window_size]
        targets = self._tokens[window_index + 1 : window_index + self._window_size + 1]
        return Datapoint(inputs, targets)

    @staticmethod
    def collate(datapoints: Sequence[Datapoint]) -> DatapointBatch:
        return DatapointBatch(
            np.stack([datapoint.inputs for datapoint in datapoints]),
            np.stack([datapoint.targets for datapoint in datapoints]),
        )


class WindowDataset2(Dataset[Datapoint]):
    def __init__(self, path: PathLike, window_size: int):
        assert window_size > 1, f"Invalid window_size: {window_size}"

        self._path = path
        self._tokens = None
        self._window_size = window_size

    def _get_tokens(self) -> NDArray[np.uint16]:
        # Lazily mmap the file with tokens
        if self._tokens is None:
            self._tokens = np.memmap(self._path, dtype="<u2", mode="r")
            assert len(self._tokens.shape) == 1 and self._tokens.dtype == np.uint16, (
                f"Invalid shape {self._tokens.shape}",
            )
        return self._tokens

    def __len__(self) -> int:
        return len(self._get_tokens()) - self._window_size - 1

    def __getitem__(self, window_index: int) -> Datapoint:
        tokens = self._get_tokens()
        inputs = tokens[window_index : window_index + self._window_size]
        targets = tokens[window_index + 1 : window_index + self._window_size + 1]
        return Datapoint(inputs, targets)

    @staticmethod
    def collate(datapoints: Sequence[Datapoint]) -> DatapointBatch:
        return DatapointBatch(
            np.stack([datapoint.inputs for datapoint in datapoints]),
            np.stack([datapoint.targets for datapoint in datapoints]),
        )


def create_data_loader(
    # tokens: NDArray[np.uint16],
    path: Path,
    batch_size: int,
    context_size: int,
    num_workers: int,
) -> DataLoader[Datapoint]:
    return DataLoader(
        # dataset=WindowDataset(tokens=tokens, window_size=context_size),
        dataset=WindowDataset2(path=path, window_size=context_size),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=WindowDataset.collate,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context=multiprocessing.get_context("spawn") if num_workers > 0 else None,
    )


def main():
    args = parse_args(
        Arguments,
        prog="train",
        add_option_string_dash_variants=DashVariant.DASH,
    )
    print(args)

    model = Transformer(
        attention_type=args.attention_type,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_embeddings=GPT2_TOKENIZER.n_vocab,
        embedding_size=args.embedding_size,
        context_size=args.context_size,
        dtype=args.dtype,
    )

    key, key_weights, key_validation = jax.random.split(jax.random.key(0), num=3)
    initial_weights = jax.jit(model.init)(
        key_weights,
        jnp.empty((args.batch_size, args.context_size), dtype=jnp.uint16),
        deterministic=False,
    )

    weight_shapes = jax.tree_util.tree_map(lambda x: (x.shape, x.dtype), initial_weights)
    print(f"Weight shapes: {weight_shapes}")

    num_parameters = jax.tree_util.tree_reduce(
        lambda num_elems, x: num_elems + int(np.prod(x.shape)),
        tree=initial_weights,
        initializer=0,
    )
    print(f"Model has {num_parameters} parameters")

    learning_rate_schedule = optax.join_schedules(
        schedules=(
            optax.linear_schedule(
                init_value=0.0,
                end_value=args.learning_rate,
                transition_steps=args.num_warmup_iters,
            ),
            optax.cosine_decay_schedule(
                init_value=args.learning_rate,
                decay_steps=max(0, args.num_iters - args.num_warmup_iters),
                alpha=0.1,
            ),
        ),
        boundaries=[args.num_warmup_iters],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip),
        optax.adamw(learning_rate=learning_rate_schedule, weight_decay=args.weight_decay),
    )

    print("Creating train state")
    model_apply_fn = jax.jit(model.apply)
    # sample_fn = jax.jit(partial(sample, apply_fn=model_apply_fn))
    sample_fn = jax.jit(sample, static_argnames=["apply_fn", "sample_size", "num_samples"])
    # sample_fn = sample
    train_step_fn = jax.jit(train_step)

    train_state = TrainState.create(
        apply_fn=model_apply_fn,
        weights=freeze(initial_weights),
        optimizer=optimizer,
    )
    # We access the model weights only via train_state
    del initial_weights

    # Mmap train and validation data and create data loaders
    print("Mmaping training data")
    # train_mmap = np.memmap("datasets/tiny-stories/train.bin", dtype="<u2", mode="r")
    train_data = iter(
        create_data_loader(
            # tokens=train_mmap,
            path=Path("datasets/tiny-stories/train.bin"),
            context_size=args.context_size,
            batch_size=args.batch_size,
            num_workers=args.num_loader_workers,
        )
    )

    # validation_mmap = np.memmap("datasets/tiny-stories/val.bin", dtype="<u2", mode="r")
    validation_data = iter(
        create_data_loader(
            # tokens=validation_mmap,
            path=Path("datasets/tiny-stories/val.bin"),
            context_size=args.context_size,
            batch_size=args.batch_size,
            num_workers=args.num_loader_workers,
        )
    )

    # Init wandb
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=dataclasses.asdict(args),
        )

    checkpoint_options = orbax.checkpoint.CheckpointManagerOptions(
        create=True,
        max_to_keep=2,
    )
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        directory=Path("checkpoints").absolute(),
        item_names=("state", "metadata"),
        options=checkpoint_options,
    )

    start_time = time.time()
    print("Starting training loop")
    for num_iter in range(args.num_iters):
        key, key_train_batch = jax.random.split(key, num=2)

        batch = next(train_data)
        train_state, train_loss = train_step_fn(train_state, batch.inputs, batch.targets)

        end_time = time.time()
        if num_iter % 20 == 0:
            took_ms = int(1000.0 * (end_time - start_time))

            val_loss, generations_table = None, None
            if num_iter % 200 == 0:
                key_validation, key_validation_batch, key_sample = jax.random.split(
                    key_validation, num=3
                )

                val_batch = next(validation_data)
                (_, val_loss) = train_state.apply_fn(
                    train_state.weights,
                    indices=val_batch.inputs,
                    targets=val_batch.targets,
                    deterministic=True,
                )

                sample_indices = sample_fn(
                    apply_fn=model_apply_fn,
                    weights=train_state.weights,
                    num_samples=args.num_samples,
                    sample_size=args.sample_size,
                    key=key_sample,
                )

                generations = GPT2_TOKENIZER.decode_batch(sample_indices.tolist())
                print("\n".join(generations))
                generations_table = wandb.Table(
                    columns=["generation"],
                    data=[[generation] for generation in generations],
                )

                checkpoint_manager.save(
                    num_iter,
                    args=orbax.checkpoint.args.Composite(
                        state=orbax.checkpoint.args.StandardSave(train_state),
                        metadata=orbax.checkpoint.args.JsonSave(dataclasses.asdict(args)),
                    ),
                )
                # checkpoint_manager.wait_until_finished()

            learning_rate = float(learning_rate_schedule(num_iter))
            metrics = {
                "num_iter": num_iter,
                "num_tokens": num_iter * args.batch_size * args.context_size,
                "train/loss": train_loss,
                "learning_rate": learning_rate,
                # "mfu": running_mfu*100, # convert to percentage
                "grad_norm": float(train_state.gradient_norm),
            }
            if val_loss is not None:
                metrics["val/loss"] = float(val_loss)
            if generations_table is not None:
                metrics["generations"] = generations_table

            print(
                f"[step {num_iter}] loss: {metrics['train/loss']:>6.2}     "
                f"grad_norm: {float(train_state.gradient_norm):>6.3}     "
                f"lr: {learning_rate:>6.3}     "
                f"({took_ms:>8}ms)"
            )

            if args.wandb:
                wandb.log(metrics)

        start_time = end_time
