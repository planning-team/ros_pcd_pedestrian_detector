import torch
import fire

from model import Unet as models


INPUT_SHAPE_P = (4, 1, 16, 384)
INPUT_SHAPE_N = (4, 8, 16, 384)

INPUT_NAMES = ["p", "n"]
OUTPUT_NAMES = ["output"]


def main(weights_path: str = "/storage/nn_weights/pcd_pedestrian_detector/UNet_best.pth", 
         output_path: str = "./Unet_best.onnx"):
    model = models.ResNetUNet(1)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    _ = model.eval()
    
    dummy_p = torch.randn(INPUT_SHAPE_P)[None, 0:4, ...]
    dummy_n = torch.randn(INPUT_SHAPE_N)[None, 0:3, ...]
    dummy_input = (dummy_p, dummy_n)
    
    torch.onnx.export(model,
                      dummy_input,
                      str(output_path),
                      export_params=True,
                      verbose=False,
                      input_names=INPUT_NAMES,
                      output_names=OUTPUT_NAMES)


if __name__ == "__main__":
    fire.Fire(main)
