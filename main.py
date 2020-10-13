import tensorflow as tf
import numpy as np
from collections import OrderedDict
import glob
import os
import argparse


from pointnet import PointNetModel

# noinspection PyTypeChecker
DATA_DIR = None  # type: str
# noinspection PyTypeChecker
WEIGHTS_SAVE_PATH = None  # type: str
NUM_CLASSES = 40  # type: int
NUM_POINTS_TO_SAMPLE = 256  # type: int
NUM_EPOCHS = 10000
BATCH_SIZE = 32

MODELNET40_DOWNLOAD_URL = "http://modelnet.cs.princeton.edu/ModelNet40.zip"


def download_maybe(output_dir: str):
    data_dir = os.path.join(output_dir, "ModelNet40")
    if not os.path.exists(os.path.join(output_dir, "ModelNet40")):
        tf.keras.utils.get_file(
            "modelnet40.zip",
            MODELNET40_DOWNLOAD_URL,
            extract=True,
            cache_subdir=output_dir
        )
    return data_dir


def get_data(data_dir: str):
    global NUM_POINTS_TO_SAMPLE

    def read_OFF_file(path, y):
        raw = tf.io.read_file(path)  # Read the data.

        raw = tf.strings.substr(raw, 3, tf.strings.length(raw))  # Substring and remove the "OFF"
        raw = tf.strings.strip(raw)  # Strip the extra spaces

        raw = tf.strings.regex_replace(raw, r"#.*\n$", "\n")  # Remove comments
        raw = tf.strings.split(raw, '\n')  # Split by lines.

        meta_data = tf.strings.to_number(input=tf.strings.split(raw[0], " "), out_type=tf.int32)
        num_verts, num_faces, num_edges = tf.split(meta_data, 3)
        num_verts = tf.reshape(num_verts, ())
        # num_faces = tf.reshape(num_faces, ())
        # num_edges = tf.reshape(num_edges, ())  # Irrelevant in 'OFF' format

        start_idx_of_verts = 1  # First line is "OFF" (we removed it), second line is: "num_verts num_faces num_edges"
        end_idx_of_verts = start_idx_of_verts + num_verts
        # start_idx_of_faces = end_idx_of_verts
        # end_idx_of_faces = start_idx_of_faces + num_faces
        # start_idx_of_edges = end_idx_of_faces  # Irrelevant in 'OFF' format
        # end_idx_of_edges = start_idx_of_edges + num_edges  # Irrelevant in 'OFF' format

        vertices_raw = tf.strings.strip(raw[start_idx_of_verts:end_idx_of_verts])  # Remove extra spaces ' '
        # faces_raw = tf.strings.strip(raw[start_idx_of_faces:end_idx_of_faces])
        # edges_raw = tf.strings.strip(raw[start_idx_of_edges:end_idx_of_edges])  # Irrelevant in 'OFF' format

        points = tf.strings.to_number(tf.strings.split(vertices_raw, " "), out_type=tf.float32).to_tensor()
        points = points - tf.reduce_mean(points, axis=0, keepdims=True)

        # faces = tf.strings.to_number(tf.strings.split(faces_raw, " "), out_type=tf.int32).to_tensor()
        # _, faces = tf.split(faces, axis=1,
        #                     num_or_size_splits=(1, 3))  # Discard the first column which describes how many points.

        # edges = tf.strings.to_number(tf.strings.split(edges_raw, " "), out_type=tf.int32).to_tensor()  # Irrelevant in 'OFF' format

        if num_verts < NUM_POINTS_TO_SAMPLE:
            _zeros = tf.zeros(shape=[NUM_POINTS_TO_SAMPLE - num_verts, 3], dtype=tf.float32)
            points = tf.concat((points, _zeros), axis=0)
            points = tf.random.shuffle(points)
        else:
            points = tf.random.shuffle(points)
            points = points[:NUM_POINTS_TO_SAMPLE]
        return tf.reshape(points, [NUM_POINTS_TO_SAMPLE, 3]), y

    def augment(points, y):
        points += tf.random.uniform(shape=tf.shape(points), minval=-0.01, maxval=0.01)
        return points, y

    class_map = OrderedDict()
    folders = sorted(glob.glob(os.path.join(data_dir, "*")))  # Sorting is important!!!

    train_data_paths = []
    test_data_paths = []
    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))
        train_data_paths.append(list(train_files))
        test_data_paths.append(list(test_files))

    train_datasets = []
    for i, x in enumerate(train_data_paths):
        ds = tf.data.Dataset.from_tensor_slices((x, tf.fill([len(x)], value=i)))
        ds = ds.shuffle(buffer_size=len(x), reshuffle_each_iteration=True)
        ds = ds.repeat(count=1)
        train_datasets.append(ds)
    train_ds = tf.data.experimental.sample_from_datasets(tuple(train_datasets), weights=np.ones([len(train_datasets)]))
    train_ds = train_ds.map(read_OFF_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=512)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(1)

    test_datasets = []
    for i, x in enumerate(test_data_paths):
        ds = tf.data.Dataset.from_tensor_slices((x, tf.fill([len(x)], value=i)))
        ds = ds.repeat(count=1)
        test_datasets.append(ds)
    test_ds = test_datasets[0]
    for i in range(1, len(test_datasets)):
        test_ds = test_ds.concatenate(test_datasets[i])
    test_ds = test_ds.map(read_OFF_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.prefetch(1)

    return train_ds, test_ds


def get_model(weights_save_path: str):
    global NUM_CLASSES
    initial_epoch = 0
    best_categorical_acc = 0.0
    keras_model = PointNetModel(num_classes=NUM_CLASSES)
    weights_path = tf.train.latest_checkpoint(weights_save_path)
    if weights_path is not None:
        _, _initial_epoch_str, _best_categorical_acc_str = os.path.basename(weights_path).split('-')
        initial_epoch = int(_initial_epoch_str)
        best_categorical_acc = float(_best_categorical_acc_str)
        keras_model.load_weights(weights_path)
        print(f"Resuming from {weights_path}")
    return keras_model, initial_epoch, best_categorical_acc


def parse_args():
    global WEIGHTS_SAVE_PATH, DATA_DIR
    _default_output_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=_default_output_dir, help="Where to save the outputs")
    args = parser.parse_args()
    args.output = os.path.join(args.output, "output")
    os.makedirs(args.output, exist_ok=True)
    WEIGHTS_SAVE_PATH = os.path.join(args.output, "weights")
    DATA_DIR = os.path.join(args.output, "data")
    os.makedirs(WEIGHTS_SAVE_PATH, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    return args


def main():
    global WEIGHTS_SAVE_PATH, DATA_DIR, BATCH_SIZE, NUM_EPOCHS
    args = parse_args()
    DATA_DIR = download_maybe(output_dir=DATA_DIR)

    train_dataset, test_dataset = get_data(data_dir=DATA_DIR)

    # Creating and running the model
    model, initial_epoch, best_acc = get_model(weights_save_path=WEIGHTS_SAVE_PATH)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
                           tf.keras.metrics.SparseCategoricalAccuracy()],
                  run_eagerly=False)

    model_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(WEIGHTS_SAVE_PATH, "weights-{epoch:04d}-{val_categorical_accuracy:.4f}"),
        save_best_only=True,
        monitor="val_categorical_accuracy")
    callbacks = [
        model_ckpt_callback,
        tf.keras.callbacks.ReduceLROnPlateau(verbose=1, patience=50, min_delta=0.001)
    ]

    model_ckpt_callback.best = best_acc
    model.fit(x=train_dataset, validation_data=test_dataset, epochs=NUM_EPOCHS,
              callbacks=callbacks, initial_epoch=initial_epoch)


if __name__ == '__main__':
    main()