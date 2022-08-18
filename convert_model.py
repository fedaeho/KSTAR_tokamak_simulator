from pathlib import Path

import click
import onnx
import tf2onnx
from keras import layers, models


def load_custom_model(input_shape, lstms, denses, model_path):
    model = models.Sequential()
    model.add(layers.BatchNormalization(input_shape=input_shape))
    for i, n in enumerate(lstms):
        rs = False if i == len(lstms) - 1 else True
        model.add(layers.LSTM(n, return_sequences=rs))
        model.add(layers.BatchNormalization())
    for n in denses[:-1]:
        model.add(layers.Dense(n, activation="sigmoid"))
        model.add(layers.BatchNormalization())
    model.add(layers.Dense(denses[-1], activation="linear"))
    model.load_weights(model_path)
    return model


def tf2onnx_convert_by_path(input_path, output_path):
    if "lstm" in str(input_path):
        if "v220505" in str(input_path):
            tf_model = load_custom_model(
                (10, 18), [100, 100], [50, 4], input_path
            )
        else:
            tf_model = load_custom_model(
                (10, 21), [200, 200], [200, 4], input_path
            )
    else:
        tf_model = models.load_model(input_path, compile=False)
    tf2onnx.convert.from_keras(tf_model, output_path=output_path)


def test_onnx_model(path):
    # Load the ONNX model
    model = onnx.load(path)
    # Check that the model is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(model.graph))


@click.command()
@click.option("--test", is_flag=True)
def main(test):
    WEIGHTS_ROOT = Path("./weights")
    ONNX_ROOT = Path("./models")

    WEIGHTS_PATH = [p for p in WEIGHTS_ROOT.glob("**/best*")]
    ONNX_PATH = [
        ONNX_ROOT.joinpath(p.relative_to(WEIGHTS_ROOT)).with_suffix(".onnx")
        for p in WEIGHTS_PATH
    ]

    for input_path, output_path in zip(WEIGHTS_PATH, ONNX_PATH):
        print(input_path, output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tf2onnx_convert_by_path(input_path, output_path)

        if test:
            test_onnx_model(output_path)


if __name__ == "__main__":
    main()
