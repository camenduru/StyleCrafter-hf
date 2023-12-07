import gradio as gr
import os
import sys
import argparse
import random
import time
from omegaconf import OmegaConf
import torch
import torchvision
from pytorch_lightning import seed_everything
from huggingface_hub import hf_hub_download
from einops import repeat
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from utils.utils import instantiate_from_config
from PIL import Image

from collections import OrderedDict

sys.path.insert(0, "scripts/evaluation")
from lvdm.models.samplers.ddim import DDIMSampler, DDIMStyleSampler


def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
    else:       
        # deepspeed
        state_dict = OrderedDict()
        for key in state_dict['module'].keys():
            state_dict[key[16:]]=state_dict['module'][key]

    model.load_state_dict(state_dict, strict=False)
    print('>>> model checkpoint loaded.')
    return model


def download_model():
    REPO_ID = 'VideoCrafter/Text2Video-512'
    filename_list = ['model.ckpt']
    os.makedirs('./checkpoints/videocrafter_t2v_320_512/', exist_ok=True)
    for filename in filename_list:
        local_file = os.path.join('./checkpoints/videocrafter_t2v_320_512/', filename)
        if not os.path.exists(local_file):
            hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir='./checkpoints/videocrafter_t2v_320_512/', force_download=True)

    REPO_ID = 'liuhuohuo/StyleCrafter'
    filename_list = ['adapter_v1.pth', 'temporal_v1.pth']
    os.makedirs('./checkpoints/stylecrafter', exist_ok=True)
    for filename in filename_list:
        local_file = os.path.join('./checkpoints/stylecrafter', filename)
        if not os.path.exists(local_file):
            hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir='./checkpoints/stylecrafter', force_download=True)
    

def infer(image, prompt, infer_type='image', seed=123, style_strength=1.0, steps=50):
    download_model()
    ckpt_path = 'checkpoints/videocrafter_t2v_320_512/model.ckpt'
    adapter_ckpt_path = 'checkpoints/stylecrafter/adapter_v1.pth'
    temporal_ckpt_path = 'checkpoints/stylecrafter/temporal_v1.pth'
    if infer_type == 'image':
        config_file='configs/inference_image_512_512.yaml'
        h, w = 512 // 8, 512 // 8
        unconditional_guidance_scale = 7.5
        unconditional_guidance_scale_style = None
    else:
        config_file='configs/inference_video_320_512.yaml'
        h, w = 320 // 8, 512 // 8
        unconditional_guidance_scale = 15.0
        unconditional_guidance_scale_style = 7.5

    config = OmegaConf.load(config_file)
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['adapter_config']['params']['scale'] = style_strength


    model = instantiate_from_config(model_config)
    model = model.cuda()

    # load ckpt
    assert os.path.exists(ckpt_path), "Error: checkpoint Not Found!"
    assert os.path.exists(adapter_ckpt_path), "Error: adapter checkpoint Not Found!"
    assert os.path.exists(temporal_ckpt_path), "Error: temporal checkpoint Not Found!"
    model = load_model_checkpoint(model, ckpt_path)
    model.load_pretrained_adapter(adapter_ckpt_path)
    if infer_type == 'video':
        model.load_pretrained_temporal(temporal_ckpt_path)
    model.eval()


    seed_everything(seed)

    batch_size=1
    channels = model.channels
    frames = model.temporal_length if infer_type == 'video' else 1
    noise_shape = [batch_size, channels, frames, h, w]

    # text cond
    cond = model.get_learned_conditioning([prompt])
    neg_prompt = batch_size * [""]
    uc = model.get_learned_conditioning(neg_prompt)

    # style cond
    style_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(512),
        torchvision.transforms.CenterCrop(512),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x * 2. - 1.),
    ])

    image = Image.fromarray(image.astype('uint8'), 'RGB')
    style_img = style_transforms(image).unsqueeze(0).cuda()
    style_cond = model.get_batch_style(style_img)
    append_to_context = model.adapter(style_cond)

    scale_scalar = model.adapter.scale_predictor(torch.concat([append_to_context, cond], dim=1))

    ddim_sampler = DDIMSampler(model) if infer_type == 'image' else DDIMStyleSampler(model) 
    
    samples, _ = ddim_sampler.sample(S=steps,
                                    conditioning=cond,
                                    batch_size=noise_shape[0],
                                    shape=noise_shape[1:],
                                    verbose=False,
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_guidance_scale_style=unconditional_guidance_scale_style,
                                    unconditional_conditioning=uc,
                                    eta=1.0,
                                    temporal_length=noise_shape[2],
                                    append_to_context=append_to_context,
                                    scale_scalar=scale_scalar
                                    )
    samples = model.decode_first_stage(samples)  

    if infer_type == 'image':
        samples = samples[:, :, 0, :, :].detach().cpu()
        out_path = "./output.png"
        torchvision.utils.save_image(samples, out_path, nrow=1, normalize=True, range=(-1, 1))

    elif infer_type == 'video':
        samples = samples.detach().cpu()
        out_path = "./output.mp4"
        video = torch.clamp(samples, -1, 1)
        video = video.permute(2, 0, 1, 3, 4) # [T, B, C, H, W]
        frame_grids = [torchvision.utils.make_grid(video[t], nrow=1) for t in range(video.shape[0])]
        grid = torch.stack(frame_grids, dim=0)
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).permute(0, 2, 3, 1).numpy().astype('uint8')
        torchvision.io.write_video(out_path, grid, fps=8, video_codec='h264', options={'crf': '10'})
        

    return out_path


def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content


demo_exaples_image = [
    ['eval_data/3d_1.png', 'A bouquet of flowers in a vase.', 'image', 123, 1.0, 50],
    ['eval_data/craft_1.jpg', 'A modern cityscape with towering skyscrapers.', 'image', 124, 1.0, 50],
    ['eval_data/digital_art_2.jpeg', 'A lighthouse standing tall on a rocky coast.', 'image', 123, 1.0, 50],
    ['eval_data/oil_paint_2.jpg', 'A man playing the guitar on a city street.', 'image', 123, 1.0, 50],
]
demo_exaples_video = [
    ['eval_data/craft_2.png', 'City street at night with bright lights and busy traffic.', 'video', 123, 1.0, 50],
    ['eval_data/anime_1.jpg', 'A field of sunflowers on a sunny day.', 'video', 123, 1.0, 50],
    ['eval_data/ink_2.jpeg', 'A knight riding a horse through a field.', 'video', 123, 1.0, 50],
    ['eval_data/oil_paint_2.jpg', 'A street performer playing the guitar.', 'video', 121, 1.0, 50],
    ['eval_data/icon_1.png', 'A campfire surrounded by tents.', 'video', 123, 1.0, 50],
]
css = """
#input_img {max-height: 320px; max-width: 512px;} 
#input_img [data-testid="image"], #input_img [data-testid="image"] > div{max-height: 320px; max-width: 512px;}
#output_img {max-height: 512px; max-width: 512px;}
#output_vid {max-height: 320px; max-width: 512px;}
"""

with gr.Blocks(analytics_enabled=False, css=css) as demo_iface:
    gr.HTML(read_content("header.html"))
    
    with gr.Tab(label='Stylized Image Generation'):
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_style_ref = gr.Image(label="Style Reference",elem_id="input_img")
                    with gr.Row():
                        input_prompt = gr.Text(label='Prompts')
                    with gr.Row():
                        input_seed = gr.Slider(label='Random Seed', minimum=0, maximum=1000, step=1, value=123)
                        input_style_strength = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, label='Style Strength', value=1.0)
                    with gr.Row():
                        input_step = gr.Slider(minimum=1, maximum=75, step=1, elem_id="i2v_steps", label="Sampling steps", value=50)
                        input_type = gr.Radio(choices=["image"], label="Generation Type", value="image")
                    input_end_btn = gr.Button("Generate")
                # with gr.Tab(label='Result'):
                with gr.Row():
                    output_result = gr.Image(label="Generated Results",elem_id="output_img", show_share_button=True)

            gr.Examples(examples=demo_exaples_image,
                        inputs=[input_style_ref, input_prompt, input_type, input_seed, input_style_strength, input_step],
                        outputs=[output_result],
                        fn = infer,
            )
        input_end_btn.click(inputs=[input_style_ref, input_prompt, input_type, input_seed, input_style_strength, input_step],
                        outputs=[output_result],
                        fn = infer
        )

    with gr.Tab(label='Stylized Video Generation'):
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_style_ref = gr.Image(label="Style Reference",elem_id="input_img")
                    with gr.Row():
                        input_prompt = gr.Text(label='Prompts')
                    with gr.Row():
                        input_seed = gr.Slider(label='Random Seed', minimum=0, maximum=1000, step=1, value=123)
                        input_style_strength = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, label='Style Strength', value=1.0)
                    with gr.Row():
                        input_step = gr.Slider(minimum=1, maximum=75, step=1, elem_id="i2v_steps", label="Sampling steps", value=50)
                        input_type = gr.Radio(choices=["video"], label="Generation Type", value="video")
                    input_end_btn = gr.Button("Generate")
                # with gr.Tab(label='Result'):
                with gr.Row():
                    output_result = gr.Video(label="Generated Results",elem_id="output_vid",autoplay=True,show_share_button=True)

            gr.Examples(examples=demo_exaples_video,
                        inputs=[input_style_ref, input_prompt, input_type, input_seed, input_style_strength, input_step],
                        outputs=[output_result],
                        fn = infer,
            )
        input_end_btn.click(inputs=[input_style_ref, input_prompt, input_type, input_seed, input_style_strength, input_step],
                        outputs=[output_result],
                        fn = infer
        )

demo_iface.queue(max_size=12).launch(show_api=True)