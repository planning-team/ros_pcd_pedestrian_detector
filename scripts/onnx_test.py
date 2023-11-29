import numpy as np
import torch
import fire
import time
import onnxruntime as onnxrt

from tools import preprocess
from tools import dataloader_tools as data_loader
from model import Unet as models


INPUT_SHAPE_P = (4, 1, 16, 384)
INPUT_SHAPE_N = (4, 8, 16, 384)

INPUT_NAMES = ["p", "n"]
OUTPUT_NAMES = ["output"]


def main(weights_path: str = "/storage/nn_weights/pcd_pedestrian_detector/UNet_best.pth", 
         onnx_path: str = "./Unet_best.onnx"):
    
    arrays = np.load("/storage/factory/pedestrian_detector_ws/points.npz")
    xyz = arrays["xyz"]
    xyzp_original = arrays["xyzp"]
    
    model = models.ResNetUNet(1)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model = model.to("cuda:0")
    _ = model.eval()


    pcd2img = preprocess.Pcd2ImageTransform().fit_fast(xyz)
    data = pcd2img.transform_fast()
    # data = data_loader.interp_data(data, data[:,:,3] != 0)
    p, n = data_loader.pointnetize(data[:,:,0:4], n_size=[3,3])
    p = torch.tensor(p, dtype=torch.float).permute(-1, -2, 0, 1).to("cuda:0")
    n = torch.tensor(n, dtype=torch.float).permute(-1, -2, 0, 1).to("cuda:0")
    
    start_time = time.time()
    with torch.no_grad():
        pred_model = model(p[None, 0:4, ...], n[None, 0:3, ...]).sigmoid().clone().detach().cpu().numpy()
    end_time = time.time()
    print(f"Time model: {end_time - start_time}")
    # xyzp_model = pcd2img.inverse_transform((pred.detach().cpu().numpy()[0, 0, :, :] > 0.5).astype(np.float32)).copy()
    
    pcd2img = preprocess.Pcd2ImageTransform().fit_fast(xyz)
    data = pcd2img.transform_fast()
    # data = data_loader.interp_data(data, data[:,:,3] != 0)
    p, n = data_loader.pointnetize(data[:,:,0:4], n_size=[3,3])
    p = p.transpose(-1, -2, 0, 1)[None, 0:4, ...].astype(np.float32)
    n = n.transpose(-1, -2, 0, 1)[None, 0:3, ...].astype(np.float32)
    
    onnx_session= onnxrt.InferenceSession("Unet_best.onnx")
    start_time = time.time()
    onnx_inputs= {onnx_session.get_inputs()[0].name: p,
                  onnx_session.get_inputs()[1].name: n}
    pred_onnx = onnx_session.run(None, onnx_inputs)[0]
    pred_onnx = 1. / (1 + np.exp(-pred_onnx))
    end_time = time.time()
    print(f"Time ONNX: {end_time - start_time}")
    
    
    print(pred_model.shape)
    assert np.allclose(pred_model, pred_onnx, atol=1e-4)
    
    
    # dummy_p = torch.randn(INPUT_SHAPE_P)[None, 0:4, ...]
    # dummy_n = torch.randn(INPUT_SHAPE_N)[None, 0:3, ...]
    # dummy_input = (dummy_p, dummy_n)
    
    # torch.onnx.export(model,
    #                   dummy_input,
    #                   str(output_path),
    #                   export_params=True,
    #                   verbose=False,
    #                   input_names=INPUT_NAMES,
    #                   output_names=OUTPUT_NAMES)


if __name__ == "__main__":
    fire.Fire(main)
