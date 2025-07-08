import glob
import logging
import os
from copy import deepcopy
from tqdm import tqdm
import gradio as gr
import numpy as np
import PIL
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from transformers import AutoModelForImageSegmentation
from utils import extract_object, resize_and_center_crop

from lbm.inference import get_model

PATH = os.path.dirname(os.path.abspath(__file__))
# os.environ["GRADIO_TEMP_DIR"] = ".gradio"
INPUT_FG_DIR=os.path.join(PATH,"examples/foregrounds")
INPUT_BG_DIR=os.path.join(PATH,"examples/backgrounds")
OUTPUT_RESULT_DIR=os.path.join(PATH,"examples/results")
os.makedirs(OUTPUT_RESULT_DIR,exist_ok=True)
model_dir = os.path.join(PATH, "ckpts", "relighting")

ASPECT_RATIOS = {
    str(512 / 2048): (512, 2048),
    str(1024 / 1024): (1024, 1024),
    str(2048 / 512): (2048, 512),
    str(896 / 1152): (896, 1152),
    str(1152 / 896): (1152, 896),
    str(512 / 1920): (512, 1920),
    str(640 / 1536): (640, 1536),
    str(768 / 1280): (768, 1280),
    str(1280 / 768): (1280, 768),
    str(1536 / 640): (1536, 640),
    str(1920 / 512): (1920, 512),
}

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
).cuda()
image_size = (1024, 1024)

if not os.path.exists(os.path.join(PATH, "examples")):
    logging.info(f"Downloading backgrounds from HF hub...")
    _ = snapshot_download(
        "jasperai/LBM_relighting",
        repo_type="space",
        allow_patterns="*.jpg",
        local_dir=PATH,
    )

if not os.path.exists(model_dir):
    logging.info(f"Downloading relighting LBM model from HF hub...")
    model = get_model(
        f"jasperai/LBM_relighting",
        save_dir=model_dir,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
else:
    logging.info(f"Loading relighting LBM model from local...")
    model = get_model(
        model_dir,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )



def process_image_pair(
    fg_path,
    bg_path,
    output_path,
    num_sampling_steps: int = 1,
):
    try:
        #加载图片
        fg_image = Image.open(fg_path).convert("RGB")
        bg_image = Image.open(bg_path).convert("RGB")
        ori_h_bg, ori_w_bg = fg_image.size
        logging.info(f"height:{ori_h_bg},weight:{ori_w_bg}")
        ar_bg = ori_h_bg / ori_w_bg
        closest_ar_bg = min(ASPECT_RATIOS, key=lambda x: abs(float(x) - ar_bg))
        dimensions_bg = ASPECT_RATIOS[closest_ar_bg]

        _, fg_mask = extract_object(birefnet, deepcopy(fg_image))

        fg_image = resize_and_center_crop(fg_image, dimensions_bg[0], dimensions_bg[1])
        fg_mask = resize_and_center_crop(fg_mask, dimensions_bg[0], dimensions_bg[1])
        bg_image = resize_and_center_crop(bg_image, dimensions_bg[0], dimensions_bg[1])

        img_pasted = Image.composite(fg_image, bg_image, fg_mask)

        img_pasted_tensor = ToTensor()(img_pasted).unsqueeze(0) * 2 - 1
        batch = {
            "source_image": img_pasted_tensor.cuda().to(torch.bfloat16),
        }

        z_source = model.vae.encode(batch[model.source_key])

        output_image = model.sample(
            z=z_source,
            num_steps=num_sampling_steps,
            conditioner_inputs=batch,
            max_samples=1,
        ).clamp(-1, 1)

        output_image = (output_image[0].float().cpu() + 1) / 2
        output_image = ToPILImage()(output_image)

        # paste the output image on the background image
        output_image = Image.composite(output_image, bg_image, fg_mask)

        output_image.resize((ori_h_bg, ori_w_bg))
        output_image.save(output_path)
        return True
    except Exception as e:
        logging.error("process image error")
        return False

def batch_process_images(num_sampling_steps:int = 1):
    fg_extensions = ["*.jpg","*.png","*.jpeg","*.bmp"]
    bg_extensions = ["*.jpg","*.png","*.jpeg","*.bmp"]
    fg_paths=[]
    bg_paths=[]
    for ext in fg_extensions:
        fg_paths.extend(glob.glob(os.path.join(INPUT_FG_DIR,ext)))
    for ext in bg_extensions:
        bg_paths.extend(glob.glob(os.path.join(INPUT_BG_DIR,ext)))
    
    if not fg_paths or not bg_paths:
        logging.error("No input image found, please add more samploe images.")
    logging.info(f"Found {len(fg_paths)} foreground images and {len(bg_paths)} background images")
    total_pairs= len(fg_paths)*len(bg_paths)
    logging.info(f"Total {total_pairs} image pairs to process.")

    #处理每对照片
    progress = tqdm(total=total_pairs,desc="processing images")
    success_count=0

    for fg_path in fg_paths:
        fg_name = os.path.splitext(os.path.basename(fg_path))[0]
        for bg_path in bg_paths:
            bg_name = os.path.splitext(os.path.basename(bg_path))[0]
            output_name = f"{fg_name}_relight_{bg_name}.jpg"
            output_path=os.path.join(OUTPUT_RESULT_DIR,output_name)
            if process_image_pair(fg_path, bg_path, output_path,num_sampling_steps):
                success_count+=1
            else :
                return
            progress.update(1)
    progress.close()
    logging.info(f"Progress completed.{success_count}/{total_pairs}pairs succeed.")
    logging.info(f"Results are saved in {OUTPUT_RESULT_DIR}")

# 主函数
def main():
    """主流程：加载模型并批量处理图片"""
    logging.info("Starting LBM image relighting batch processing...")
    
    # 批量处理图片
    batch_process_images(num_sampling_steps=1)  # 可调整推理步数
    
    logging.info("All tasks completed.")

if __name__ == "__main__":
    main()
