import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import FluxPipeline
from PIL import Image
import argparse
import random 
import numpy as np
import yaml
import os
from FlowEdit_utils import FlowEditSD3, FlowEditFLUX



if __name__ == "__main__":

    # 1. Argument Parsing:
    parser = argparse.ArgumentParser()
    # Specifies the GPU device to use.
    parser.add_argument("--device_number", type=int, default=0, help="device number to use")
    # Path to the YAML file containing experiment configurations.
    parser.add_argument("--exp_yaml", type=str, default="FLUX_exp.yaml", help="experiment yaml file")

    args = parser.parse_args()

    # 2.Setting up the device: checks whether a GPU (CUDA-enabled device) is available and assigns the specified device
    device_number = args.device_number
    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")

    # load exp yaml file to dict
    exp_yaml = args.exp_yaml
    with open(exp_yaml) as file:
        exp_configs = yaml.load(file, Loader=yaml.FullLoader)

    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")
    model_type = exp_configs[0]["model_type"] # currently only one model type per run

    # 4. Loading Models (FLUX or SD3):

    if model_type == 'FLUX':
        # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16) 
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
    elif model_type == 'SD3':
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
    
    scheduler = pipe.scheduler
    pipe = pipe.to(device)


    # 3. Loading Experiment Configurations - The experiment configurations are loaded from the YAML file. It contains multiple settings like:
    for exp_dict in exp_configs:

        exp_name = exp_dict["exp_name"]
        # model_type = exp_dict["model_type"]
        T_steps = exp_dict["T_steps"]
        n_avg = exp_dict["n_avg"]
        src_guidance_scale = exp_dict["src_guidance_scale"]
        tar_guidance_scale = exp_dict["tar_guidance_scale"]
        n_min = exp_dict["n_min"]
        n_max = exp_dict["n_max"]
        seed = exp_dict["seed"]

        # set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        dataset_yaml = exp_dict["dataset_yaml"]
        with open(dataset_yaml) as file:
            dataset_configs = yaml.load(file, Loader=yaml.FullLoader)

        # check dataset_configs 
        for data_dict in dataset_configs:
            tar_prompts = data_dict["target_prompts"]


        #6. Main Transformation Loop:
        for data_dict in dataset_configs:

            src_prompt = data_dict["source_prompt"]
            tar_prompts = data_dict["target_prompts"]
            negative_prompt =  "" # optionally add support for negative prompts (SD3)

            #5. Image Preprocessing:
            #For each experiment in the loaded YAML, the script processes images by taking a source image (input_img) and transforming it based on corresponding source and target prompts:
            image_src_path = data_dict["input_img"]

            # load image
            image = Image.open(image_src_path)
            # crop image to have both dimensions divisibe by 16 - avoids issues with resizing
            image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
            image_src = pipe.image_processor.preprocess(image)
            # cast image to half precision
            image_src = image_src.to(device).half()
            with torch.autocast("cuda"), torch.inference_mode():
                x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
            x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            # send to cuda
            x0_src = x0_src.to(device)
            

            # Loading Models (FLUX or SD3)
            # Depending on the model type, it uses either FlowEditSD3 (for SD3) or FlowEditFLUX (for FLUX) to generate a target image.
            for tar_num, tar_prompt in enumerate(tar_prompts):

                if model_type == 'SD3':
                    x0_tar = FlowEditSD3(pipe,
                                                            scheduler,
                                                            x0_src,
                                                            src_prompt,
                                                            tar_prompt,
                                                            negative_prompt,
                                                            T_steps,
                                                            n_avg,
                                                            src_guidance_scale,
                                                            tar_guidance_scale,
                                                            n_min,
                                                            n_max,)
                    
                elif model_type == 'FLUX':
                    x0_tar = FlowEditFLUX(pipe,
                                                            scheduler,
                                                            x0_src,
                                                            src_prompt,
                                                            tar_prompt,
                                                            negative_prompt,
                                                            T_steps,
                                                            n_avg,
                                                            src_guidance_scale,
                                                            tar_guidance_scale,
                                                            n_min,
                                                            n_max,)
                else:
                    raise NotImplementedError(f"Sampler type {model_type} not implemented")

                # 7.Image Postprocessing and Saving:
                x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                with torch.autocast("cuda"), torch.inference_mode():
                    image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
                image_tar = pipe.image_processor.postprocess(image_tar)

                src_prompt_txt = data_dict["input_img"].split("/")[-1].split(".")[0]

                tar_prompt_txt = str(tar_num)
                
                # 8. Directory Management: make sure to create the directories before saving
                save_dir = f"outputs/{exp_name}/{model_type}/src_{src_prompt_txt}/tar_{tar_prompt_txt}"
                os.makedirs(save_dir, exist_ok=True)
                
                # 9. Output File Naming:
                image_tar[0].save(f"{save_dir}/output_T_steps_{T_steps}_n_avg_{n_avg}_cfg_enc_{src_guidance_scale}_cfg_dec{tar_guidance_scale}_n_min_{n_min}_n_max_{n_max}_seed{seed}.png")
                # also save source and target prompt in txt file
                with open(f"{save_dir}/prompts.txt", "w") as f:
                    f.write(f"Source prompt: {src_prompt}\n")
                    f.write(f"Target prompt: {tar_prompt}\n")
                    f.write(f"Seed: {seed}\n")
                    f.write(f"Sampler type: {model_type}\n")
                




    print("Done")