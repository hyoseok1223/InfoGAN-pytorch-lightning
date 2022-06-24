import gradio as gr
import numpy as np
import torch
from train import GAN
from utils import *
from torchvision.utils import save_image
import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./infogan/config.yaml', help='Base config') # do not change

    args = parser.parse_args()
    src_config_dir = os.path.join(os.getcwd(), args.config)
    cfg = load_config(src_config_dir)

    # concat args & config
    dict_args = vars(args)
    dict_args.update(cfg)
    args = argparse.Namespace(**dict_args)
    args.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args 

def infogan(discrete_num , c_continuous_1,c_continuous_2):
    z = torch.randn(1,62)
    c_discrete = to_onehot(discrete_num)
    c_continuous = torch.Tensor([[c_continuous_1,c_continuous_2]])
    c = torch.cat((c_discrete.float(), c_continuous), 1)



    model = GAN()
    model = model.load_from_checkpoint(checkpoint_path='example.ckpt')
    model.eval()
    result = model(z, c)

    save_image(result, "result.png", normalize=True)


    return "result.png"

if __name__ =='__main__':
    args = get_parser()
    torch.manual_seed(42)
    iface = gr.Interface(fn=infogan,
                        inputs=[gr.inputs.Slider(0,9,step=1),gr.inputs.Slider(-1.,1.0),gr.inputs.Slider(-1.,1.0)], 
                        outputs="image")
    iface.launch()