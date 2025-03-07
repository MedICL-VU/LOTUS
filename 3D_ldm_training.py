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

from mytransforms import *
from utiles import *

print_config()

set_determinism(42)

batch_size = 1
img_dim = 64
img_space = 2
n_epochs = 200
adv_weight = 0.01
perceptual_weight = 0.01
kl_weight = 1e-6
latent_c = 1
dataname = 'your_data_name'
train_name = "1st_"+dataname+"_bs"+str(batch_size)+"dim"+str(img_dim)+"iS"+str(img_space)+"E"+str(n_epochs)+"aW"+str(adv_weight)+"pW"+str(perceptual_weight) + 'kW'+str(kl_weight) +'lC'+str(latent_c)
start_time = time.time()
autoencoder_path = '/pretrained_autoencoder'
input_data_dir = '/data/'+dataname
save_dir = '/results/3D_LDM'+train_name
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
val_ds = CacheDataset(data = val_files, transform=train_transforms)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

autoencoder = AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 64),
    latent_channels=latent_c,
    num_res_blocks=1,
    norm_num_groups=16,
    attention_levels=(False, False, True),
)
autoencoder.to(device)
autoencoder.load_state_dict(torch.load(autoencoder_path))
autoencoder.eval()

unet = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=latent_c,
    out_channels=latent_c,
    num_res_blocks=1,
    num_channels=(32, 64, 64),
    attention_levels=(False, True, True),
    num_head_channels=(0, 64, 64),
)
unet.to(device)


scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)
check_data = first(train_loader)
with torch.no_grad():
    with autocast(enabled=True):
        z = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))

print(f"Scaling factor set to {1/torch.std(z)}")
scale_factor = 1 / torch.std(z)

inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=1e-4)

epoch_loss_list = []
autoencoder.eval()
scaler = GradScaler()
val_interval = 20
first_batch = first(train_loader)
z = autoencoder.encode_stage_2_inputs(first_batch["image"].to(device))
print(f'z shape is {z.shape}')
for epoch in range(n_epochs):
    unet.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        optimizer_diff.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn_like(z).to(device)

            # Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            # Get model prediction
            noise_pred = inferer(
                inputs=images, autoencoder_model=autoencoder, diffusion_model=unet, noise=noise, timesteps=timesteps
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer_diff)
        scaler.update()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_loss_list.append(epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        unet.eval()
        model_save_path = os.path.join(save_dir, f'ldm_epoch_{epoch + 1}.pth')
        torch.save(unet.state_dict(), model_save_path)

        noise = torch.randn_like(z)  # TODO pay attention to the noise size, can actually generate larger image!
        noise = noise.to(device)
        scheduler.set_timesteps(num_inference_steps=1000)
        synthetic_images = inferer.sample(
            input_noise=noise, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler
        )
        plt.figure(figsize=(4, 3))
        plt.subplots(1, 3, figsize=(4, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(synthetic_images[0, 0, :, img_dim // 2, :].cpu(), vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.title("sampled image")
        plt.subplot(1, 3, 2)
        plt.imshow(synthetic_images[0, 0, img_dim // 2, :, :].cpu(), vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.title("sampled image")
        plt.subplot(1, 3, 3)
        plt.imshow(synthetic_images[0, 0, :, :, img_dim // 2].cpu(), vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.title("sampled image")
        plt.savefig(os.path.join(save_vis_dir, f'val_vis_e{epoch + 1}.png'))
        plt.close()


plt.plot(epoch_loss_list)
plt.title("Learning Curves", fontsize=20)
plt.plot(epoch_loss_list)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.savefig(os.path.join(save_vis_dir, f'training curves.png'))
plt.close()