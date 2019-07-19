import mxnet as mx
import tvm
from tvm import relay
from tvm.contrib import emscripten
import numpy as np
import nnvm

if __name__ == '__main__':
    shape_dict = {'data': (1, 3, 224, 224)}
    net = mx.gluon.model_zoo.vision.get_model('resnet18_v1', pretrained=True)
    # net.hybridize()
    # net(mx.nd.zeros((1, 3, 224, 224)))
    # net.export('vgg16')
    # mx_sym, args, auxs = mx.model.load_checkpoint('vgg16', 0)
    # mod, relay_params = relay.frontend.from_mxnet(mx_sym, arg_params=args, aux_params=auxs)
    sym, relay_params = nnvm.frontend.from_mxnet(net, shape_dict)
    mod = nnvm.graph.create(sym)
    print(mod.ir())
    target = "llvm -target=asmjs-unknown-emscripten -system-lib"
    with nnvm.compiler.build_config(opt_level=3):
        graph, lib, params = nnvm.compiler.build(mod, target, shape=shape_dict, params=relay_params)
    print(lib)
    obj_path = 'wolf_mann.bc'
    lib.save(obj_path)
    emscripten.create_js("wolf_mann.js", obj_path)