import mxnet as mx
import tvm
from tvm import relay
from tvm.contrib import emscripten
import numpy as np

if __name__ == '__main__':
    shape_dict = {'data': (1, 480)}
    mx_sym, args, auxs = mx.model.load_checkpoint('wolf_mann', 0)
    mod, relay_params = relay.frontend.from_mxnet(mx_sym, shape_dict, arg_params=args, aux_params=auxs)
    print(mod)
    target = "llvm -target=asmjs-unknown-emscripten -system-lib"
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=relay_params)
    print(lib)
    obj_path = 'wolf_mann.bc'
    lib.save(obj_path)
    emscripten.create_js("wolf_mann.js", obj_path)