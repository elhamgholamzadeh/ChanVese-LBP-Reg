import argparse
import datetime
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .data import (
    example_gen,
    read_move_data,
    read_move_labels,
    read_template_images,
    read_template_labels,
)
from .losses import gradient_loss, m_loss
from .model import unet_statistic


def print_loss(step, training, train_loss):
    s = "iter:" + str(step) + "," + str(training)

    if isinstance(train_loss, (list, np.ndarray)):
        for value in train_loss:
            s += "," + str(value)
    else:
        s += "," + str(train_loss)

    print(s)
    sys.stdout.flush()


def train(
    model_name,
    data_root,
    output_dir,
    lr=1e-4,
    n_iterations=2000,
    reg_param=1.0,
    model_save_iter=100,
    save_model_name=None,
    initial_weights=None,
):
    tf.random.set_seed(1374)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    move_images = read_move_data(data_root)
    template_images = read_template_images(data_root)
    template_labels = read_template_labels(data_root)
    move_labels = read_move_labels(data_root)

    vol_size = (160, 192, 224)
    nf_enc = [16, 32, 32, 32]

    if model_name == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8, 3]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16, 3]

    model = unet_statistic(vol_size, nf_enc, nf_dec)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=[m_loss(vol_size=vol_size), gradient_loss("l2")],
        loss_weights=[1.0, reg_param],
    )

    if initial_weights:
        model.load_weights(initial_weights)

    train_example_gen = example_gen(
        move_images,
        template_images,
        move_labels,
        template_labels,
        vol_size,
    )

    zero_flow = np.zeros((1, vol_size[0], vol_size[1], vol_size[2], 3), dtype="float32")
    losses = []

    print("start", datetime.datetime.now())

    for step in range(n_iterations):
        x, y, x_label = next(train_example_gen)

        y_label = y[:, :, :, :, 1:] + 1
        y_out = np.concatenate([x[:, :, :, :, :1], x_label + 1], axis=-1)

        train_loss = model.train_on_batch(
            [x, y[:, :, :, :, :1], x_label + 1, y_label],
            y=[y_out, zero_flow],
        )

        losses.append(train_loss)
        print_loss(step, " train loss: ", train_loss)

        if model_save_iter > 0 and step > 0 and step % model_save_iter == 0:
            model.save(output_dir / f"{step}.h5")

    if save_model_name is None:
        save_model_name = f"totalloss_0.19sto_lr{lr}_E{n_iterations}"

    model.save(output_dir / f"{save_model_name}.h5")
    np.savetxt(output_dir / f"{save_model_name}.txt", np.asarray(losses, dtype=object), fmt="%s")

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.savefig(output_dir / f"{save_model_name}_loss.png", dpi=150, bbox_inches="tight")

    print(f"'{save_model_name}.h5' saved")
    print(f"'{save_model_name}.txt' saved")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train MRI registration model.")
    parser.add_argument("--model", default="vm1", choices=["vm1", "vm2"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--initial-weights", default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-iterations", type=int, default=2000)
    parser.add_argument("--reg-param", type=float, default=1.0)
    parser.add_argument("--model-save-iter", type=int, default=100)
    parser.add_argument("--save-model-name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_name=args.model,
        data_root=args.data_root,
        output_dir=args.output_dir,
        lr=args.lr,
        n_iterations=args.n_iterations,
        reg_param=args.reg_param,
        model_save_iter=args.model_save_iter,
        save_model_name=args.save_model_name,
        initial_weights=args.initial_weights,
    )
