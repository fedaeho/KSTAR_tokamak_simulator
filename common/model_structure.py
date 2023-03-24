import numpy as np
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare

# from abc import ABCmeta

# import onnx

# # Load the ONNX model
# model = onnx.load("model/lstm_0.onnx")

# # Check that the model is well formed
# onnx.checker.check_model(model)

# # Print a human readable representation of the graph
# print(onnx.helper.printable_graph(model.graph))


# self.kstar_nn = kstar_nn(model_path=nn_model_path, n_models=1)
# self.kstar_lstm = kstar_v220505(model_path=lstm_model_path, n_models=max_models)
# self.k2rz = k2rz(model_path=k2rz_model_path, n_models=max_shape_models)
# self.bpw_nn = tf_dense_model(
#     model_path = bpw_model_path,
#     n_models = max_models,
#     ymean = [1.3630552066021155, 251779.19861710534],
#     ystd = [0.6252123013157276, 123097.77805034176]
# )


class onnx_model:
    def __init__(self, model_name, backend="ort"):
        self.backend = backend
        if backend == "ort":
            self.model = ort.InferenceSession(
                model_name, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.model.get_inputs()[0].name
        elif backend == "tf":
            self.model = prepare(onnx.load(model_name))  # load onnx model

    def predict(self, inputs):
        if self.backend == "ort":
            onnx_inputs = {self.input_name: inputs.astype(np.float32)}
            onnx_outputs = self.model.run(None, onnx_inputs)[0]
        elif self.backend == "tf":
            onnx_inputs = inputs.astype(np.float32)
            onnx_outputs = self.model.run(onnx_inputs)[0]

        print(onnx_outputs)
        return onnx_outputs


# class Model(metaclass=ABCmeta):
#     @abstractmethod
#     def set_inputs(self):
#         pass

#     @abstractmethod
#     def predict(self):
#         pass


class kstar_nn:
    def __init__(self, model_path, n_models=1, ymean=None, ystd=None):
        self.nmodels = n_models
        if ymean is None:
            self.ymean = [1.22379703, 5.2361062, 1.64438005, 1.12040048]
            self.ystd = [0.72255576, 1.5622809, 0.96563557, 0.23868018]
        else:
            self.ymean, self.ystd = ymean, ystd
        self.models = [
            onnx_model(f"models/nn/best_model{i}.onnx")
            for i in range(self.nmodels)
        ]

    def set_inputs(self, x):
        self.x = np.array(x) if len(np.shape(x)) == 2 else np.array([x])

    def predict(self, x=None):
        if type(x) == type(np.zeros(1)):
            self.set_inputs(x)
        self.y = np.mean(
            [
                m.predict(self.x)[0] * self.ystd + self.ymean
                for m in self.models[: self.nmodels]
            ],
            axis=0,
        )
        return self.y


class kstar_v220505:
    def __init__(
        self, model_path, n_models=1, ymean=None, ystd=None, length=10
    ):
        if ymean is None or ystd is None:
            self.ymean = [1.4361666, 5.275876, 1.534538, 1.1268075]
            self.ystd = [0.7294007, 1.5010427, 0.6472052, 0.2331879]
        else:
            self.ymean, self.ystd = ymean, ystd
        self.nmodels = n_models
        self.models = [
            onnx_model(f"models/lstm/v220505/best_model{i}.onnx")
            for i in range(self.nmodels)
        ]

    def set_inputs(self, x):
        self.x = np.array(x) if len(np.shape(x)) == 3 else np.array([x])

    def predict(self, x=None):
        if type(x) == type(np.zeros(1)):
            self.set_inputs(x)
        self.y = np.mean(
            [
                m.predict(self.x)[0] * self.ystd + self.ymean
                for m in self.models[: self.nmodels]
            ],
            axis=0,
        )
        return self.y


class k2rz:
    def __init__(
        self,
        model_path,
        n_models=1,
        ntheta=64,
        closed_surface=True,
        xpt_correction=True,
    ):
        self.nmodels, self.ntheta = n_models, ntheta
        self.closed_surface, self.xpt_correction = (
            closed_surface,
            xpt_correction,
        )
        self.models = [
            onnx_model(f"models/k2rz/best_model{i}.onnx")
            for i in range(self.nmodels)
        ]

    def set_inputs(self, ip, bt, βp, rin, rout, k, du, dl):
        self.x = np.array([ip, bt, βp, rin, rout, k, du, dl])

    def predict(self, post=True):
        self.y = np.mean(
            [
                m.predict(np.array([self.x]))[0]
                for m in self.models[: self.nmodels]
            ],
            axis=0,
        )
        rbdry, zbdry = self.y[: self.ntheta], self.y[self.ntheta :]
        if post:
            if self.xpt_correction:
                rgeo, amin = 0.5 * (max(rbdry) + min(rbdry)), 0.5 * (
                    max(rbdry) - min(rbdry)
                )
                if self.x[6] <= self.x[7]:
                    rx = rgeo - amin * self.x[7]
                    zx = max(zbdry) - 2 * self.x[5] * amin
                    rx2 = rgeo - amin * self.x[6]
                    rbdry[np.argmin(zbdry)] = rx
                    zbdry[np.argmin(zbdry)] = zx
                    rbdry[np.argmax(zbdry)] = rx2
                else:
                    rx = rgeo - amin * self.x[6]
                    zx = min(zbdry) + 2 * self.x[5] * amin
                    rx2 = rgeo - amin * self.x[7]
                    rbdry[np.argmax(zbdry)] = rx
                    zbdry[np.argmax(zbdry)] = zx
                    rbdry[np.argmin(zbdry)] = rx2

            if self.closed_surface:
                rbdry, zbdry = np.append(rbdry, rbdry[0]), np.append(
                    zbdry, zbdry[0]
                )

        return rbdry, zbdry


class tf_dense_model:
    def __init__(self, model_path, n_models=1, ymean=0, ystd=1):
        self.nmodels = n_models
        self.ymean, self.ystd = ymean, ystd
        self.models = [
            onnx_model(f"models/bpw/v220505/best_model{i}.onnx")
            for i in range(self.nmodels)
        ]

    def set_inputs(self, x):
        self.x = np.array(x) if len(np.shape(x)) == 2 else np.array([x])

    def predict(self, x):
        self.set_inputs(x)
        self.y = np.mean(
            [
                m.predict(self.x)[0] * self.ystd + self.ymean
                for m in self.models[: self.nmodels]
            ],
            axis=0,
        )
        return self.y
