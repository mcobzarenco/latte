import dataclasses
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import flax
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tiktoken
import wandb
from flax.core.frozen_dict import FrozenDict
from jax import Array
from numpy.typing import NDArray
from optax._src.base import GradientTransformation
from simple_parsing import parse as parse_args
from simple_parsing.wrappers.field_wrapper import DashVariant

from .model import Transformer

GPT2_TOKENIZER = tiktoken.get_encoding("gpt2")


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

    # Number of layers in the tranfomer
    num_layers: int = 2

    # Number of heads in multihead attention
    num_heads: int = 4

    # Embedding size
    embedding_size: int = 256

    # Context length in number of tokens
    context_size: int = 1024

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

    # Log to wandb
    wandb: bool = False

    # Number of training iterations
    wandb_project: str = "latte"

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
    sample_length: int,
    key: Array,
):
    indices = jnp.expand_dims(jnp.repeat(jnp.array(GPT2_TOKENIZER.eot_token), num_samples), -1)

    for num_token in range(sample_length):
        (logits, _) = apply_fn(weights, indices=indices, deterministic=True)

        key, key_generation = jax.random.split(key)
        next_indices = jax.random.categorical(key_generation, logits[:, -1:, :], axis=-1)
        indices = jnp.hstack([indices, next_indices])

    print("\n".join(GPT2_TOKENIZER.decode_batch(indices.tolist())))


class TrainState(flax.struct.PyTreeNode):
    step: int
    gradient_norm: Array
    apply_fn: Callable = flax.struct.field(pytree_node=False)
    optimizer: GradientTransformation = flax.struct.field(pytree_node=False)
    weights: FrozenDict[str, Any]
    optimizer_state: optax.OptState

    def apply_gradients(self, *, grads, **kwargs):
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


@jax.jit
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


def main():
    args = parse_args(
        Arguments,
        prog="train",
        add_option_string_dash_variants=DashVariant.DASH,
    )

    model = Transformer(
        attention_type=args.attention_type,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_embeddings=GPT2_TOKENIZER.n_vocab,
        embedding_size=args.embedding_size,
        context_size=args.context_size,
    )

    key, key_weights, key_validation = jax.random.split(jax.random.key(0), num=3)
    weights = model.init(
        key_weights,
        jnp.empty((args.batch_size, args.context_size), dtype=jnp.uint16),
        deterministic=False,
    )

    weight_shapes = jax.tree_util.tree_map(lambda x: x.shape, weights)
    print(f"Weight shapes: {weight_shapes}")

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

    # data
    train_data = np.memmap("datasets/tiny-stories/train.bin", dtype="<u2", mode="r")
    validation_data = np.memmap("datasets/tiny-stories/val.bin", dtype="<u2", mode="r")

    assert isinstance(weights, FrozenDict), type(weights)
    train_state = TrainState.create(
        apply_fn=model.apply,
        weights=weights,
        optimizer=optimizer,
    )

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            # name=wandb_run_name,
            config=dataclasses.asdict(args),
        )

    model_apply_fn = jax.jit(model.apply)

    sample_train_batch = partial(
        sample_batch,
        indices=train_data,
        batch_size=args.batch_size,
        context_size=args.context_size,
    )
    sample_validation_batch = partial(
        sample_batch,
        indices=validation_data,
        batch_size=args.batch_size,
        context_size=args.context_size,
    )

    start_time = time.time()
    for num_iter in range(args.num_iters):
        key, key_train_batch = jax.random.split(key, num=2)

        (x, y) = sample_train_batch(key=key_train_batch)
        train_state, train_loss = train_step(train_state, x, y)

        end_time = time.time()
        if num_iter % 10 == 0:
            took_ms = int(1000.0 * (end_time - start_time))

            val_loss = None
            if num_iter % 200 == 0:
                key_validation, key_validation_batch, key_sample = jax.random.split(
                    key_validation, num=3
                )

                (x_val, y_val) = sample_validation_batch(key=key_validation_batch)
                (_, val_loss) = train_state.apply_fn(
                    train_state.weights,
                    indices=x_val,
                    targets=y_val,
                    deterministic=True,
                )

                sample(
                    apply_fn=model_apply_fn,
                    weights=train_state.weights,
                    num_samples=4,
                    sample_length=50,
                    key=key_sample,
                )

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

            print(
                f"[step {num_iter}] loss: {metrics['train/loss']:>6.2}     "
                f"grad_norm: {float(train_state.gradient_norm):>6.3}     "
                f"lr: {learning_rate:>6.3}     "
                f"({took_ms:>8}ms)"
            )
            if args.wandb:
                wandb.log(metrics)

        start_time = end_time
