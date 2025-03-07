import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from tqdm import tqdm

from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from mytransforms import *
from utiles import *

print_config()

set_determinism(42)

batch_size = 1
img_dim = 64
img_space = 2
mask_space = 2
n_epochs = 2000
num_step = 1000
sch_type = "DDPM"
# %% Data Loading
start_time = time.time()
dataname = 'your_data_name'
input_data_dir = '/data/'+dataname
train_name = "1st_"+dataname+"_bs"+str(batch_size)+"iD"+str(img_dim)+"iS"+str(img_space)+"mS"+str(mask_space)+"E"+str(n_epochs)+sch_type+str(num_step)
save_dir = '/results/RePaint3D/ddpm/'+train_name
save_vis_dir = os.path.join(save_dir,"vis")
os.makedirs(save_vis_dir, exist_ok=True)

img_list_pattern = list_nii_files(input_data_dir + '/train')
train_img_list = natsorted(img_list_pattern)

img_list_pattern = list_nii_files(input_data_dir + '/val')
val_img_list = natsorted(img_list_pattern)

train_files = [
    {
        "image":train_img_list[i],
    }
    for i in range(len(train_img_list))
]

val_files = [
    {
        "image":val_img_list[i],
    }
    for i in range(len(val_img_list))
]

print(f'train file is {train_files}')
print(f'val file is {val_files}')
print(f'train file length is {len(train_files)}')
print(f'val file length is {len(val_files)}')
# %% transform
channel = 0  # 0 = Flair
assert channel in [0, 1, 2, 3], "Choose a valid channel"
rot = 2
trans = 12
scale = 0.2
prob = 0.5
train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
        transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="LPI"), # TODO Here change RAS to LPI
        # transforms.Spacingd(keys=["image"], pixdim=(img_space, img_space, img_space), mode=("bilinear")),
        # transforms.RandAdjustContrastd(keys=["image"],prob=1,gamma=(0.7,1.8)),# gamma>1: the image goes to darker, < 1 brighter
        # TODO here may add an affine later
        # transforms.CenterSpatialCropd(keys=["image"], roi_size=(img_dim, img_dim, img_dim)),
        # transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        transforms.RandAffined(
            keys=["image"],
            # keys=["image"],
            rotate_range=[(-np.pi / rot, np.pi / rot), (-np.pi / rot, np.pi / rot), (-np.pi / rot, np.pi / rot)],
            translate_range=[(-trans, trans), (-trans, trans), (-trans, trans)],  # x is up and down and y is left and right translation pixel nunmber
            scale_range=[(-scale, scale), (-scale, scale),(-scale, scale)],
            spatial_size=[img_dim, img_dim, img_dim],  # change here to 128 128
            padding_mode="reflection",  # if image is out of boundary due to translation, padding with 0
            prob=prob,
            # device="cuda",
            mode = ["bilinear"]
        ),
        # transforms.RandSpatialCropd(keys=["image"], roi_size=(img_dim, img_dim, img_dim)),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=(img_dim, img_dim, img_dim)),
    ]
)
train_ds = CacheDataset(data = train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
print(f'Image shape {train_ds[0]["image"].shape}')

val_ds = CacheDataset(data = val_files, transform = train_transforms)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)

# %% define model
device = torch.device("cuda")

model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=[256, 256, 512],
    attention_levels=[False, False, True],
    num_head_channels=[0, 0, 512],
    num_res_blocks=2,
)
model.to(device)

scheduler = DDPMScheduler(num_train_timesteps=num_step, schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195)
# scheduler = DDIMScheduler(num_train_timesteps=num_step, schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195)
inferer = DiffusionInferer(scheduler)
optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-5)

# %% train
val_interval = 50
epoch_loss_list = []
val_epoch_loss_list = []

scaler = GradScaler()
total_start = time.time()
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn_like(images).to(device)

            # Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            # Get model prediction
            noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)

            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_loss_list.append(epoch_loss / (step + 1))

    if (epoch+1) % val_interval == 0:
        model.eval()
        model_save_path = os.path.join(save_dir, f'ddpm_model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_save_path)
        val_epoch_loss = 0
        for step, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            noise = torch.randn_like(images).to(device)
            with torch.no_grad():
                with autocast(enabled=True):
                    timesteps = torch.randint(
                        0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                    ).long()

                    # Get model prediction
                    noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                    val_loss = F.mse_loss(noise_pred.float(), noise.float())

            val_epoch_loss += val_loss.item()
            progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
        val_epoch_loss_list.append(val_epoch_loss / (step + 1))

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")

