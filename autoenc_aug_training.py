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
from tests.test_spade_autoencoderkl import device
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

def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(kl_loss) / kl_loss.shape[0]

batch_size = 1
img_dim = 64
img_space = 4
n_epochs = 1500
latent_c = 1
adv_weight = 0.01
perceptual_weight = 0.01
kl_weight = 1e-6
dataname = 'your_data_name'
train_name = "1st_"+dataname+"_bs"+str(batch_size)+"dim"+str(img_dim)+"iS"+str(img_space)+"E"+str(n_epochs)+"aW"+str(adv_weight)+"pW"+str(perceptual_weight) + 'kW'+str(kl_weight)+ 'lC'+str(latent_c)
start_time = time.time()
input_data_dir = '/data/'+dataname
save_dir = '/results/LOTUS/autoencoder/'+train_name
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
        transforms.Spacingd(keys=["image"], pixdim=(img_space, img_space, img_space), mode=("bilinear")),
        transforms.ForegroundMaskD(keys = ["image"], new_key_prefix = "mask", threshold = 0.999, invert = True), # so we get a key called
        transforms.ForegroundMaskD(keys=["image"], new_key_prefix="affmask", threshold=0.999, invert=True),
        # so we get a key called
        # transforms.RandAdjustContrastd(keys=["image"],prob=1,gamma=(0.7,1.8)),# gamma>1: the image goes to darker, < 1 brighter
        transforms.RandAffined(
            keys=["image", "maskimage"],
            # keys=["image"],
            rotate_range=[(-np.pi / rot, np.pi / rot), (-np.pi / rot, np.pi / rot), (-np.pi / rot, np.pi / rot)],
            translate_range=[(-trans, trans), (-trans, trans), (-trans, trans)],  # x is up and down and y is left and right translation pixel nunmber
            scale_range=[(-scale, scale), (-scale, scale),(-scale, scale)],
            spatial_size=[img_dim, img_dim, img_dim],
            padding_mode="zeros",
            prob=prob,
            # device="cuda",
            mode = ["bilinear", 'nearest']
        ),
        ## open it when you want more patch aug
        # transforms.RandAffined(
        #     keys=["affmaskimage"],
        #     # keys=["image"],
        #     rotate_range=[(-np.pi / rot, np.pi / rot), (-np.pi / rot, np.pi / rot), (-np.pi / rot, np.pi / rot)],
        #     translate_range=[(-trans, trans), (-trans, trans), (-trans, trans)],
        #     # x is up and down and y is left and right translation pixel nunmber
        #     scale_range=[(-scale, scale), (-scale, scale), (-scale, scale)],
        #     spatial_size=[img_dim, img_dim, img_dim],
        #     padding_mode="zeros",
        #     prob=prob,
        #     # device="cuda",
        #     mode=['nearest']
        # ),

        # transforms.CenterSpatialCropd(keys=["image", "maskimage"], roi_size=(img_dim, img_dim, img_dim)),
    ]
)
train_ds = CacheDataset(data = train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
print(f'Image shape {train_ds[0]["image"].shape}')
val_ds = CacheDataset(data = val_files, transform=train_transforms)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)

check_data = first(train_loader)
idx = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

autoencoder = AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 64),
    latent_channels= 1,
    num_res_blocks=1,
    norm_num_groups=16,
    attention_levels=(False, False, True),
)
autoencoder.to(device)


discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1)
discriminator.to(device)

l1_loss = L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")
loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
loss_perceptual.to(device)

optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4)
optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4)

n_epochs = n_epochs
autoencoder_warm_up_n_epochs = 1
val_interval = 10
epoch_recon_loss_list = []
epoch_gen_loss_list = []
epoch_disc_loss_list = []
val_recon_epoch_loss_list = []
intermediary_images = []
n_example_images = 4

for epoch in range(n_epochs):
    autoencoder.train()
    discriminator.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        # if random.choice(["Heads", "Tails"]) == "Heads":
        #     images = batch["image"].to(device)  # choose only one of Brats channels
        # else:
        #     images = batch["affmaskimage"]*batch["image"].to(device)
        images = batch["image"].to(device)  # choose only one of Brats channels
        # Generator part
        optimizer_g.zero_grad(set_to_none=True)
        reconstruction, z_mu, z_sigma = autoencoder(images)
        kl_loss = KL_loss(z_mu, z_sigma)

        recons_loss = l1_loss(reconstruction.float(), images.float())
        p_loss = loss_perceptual(reconstruction.float(), images.float())
        loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

        if epoch > autoencoder_warm_up_n_epochs:
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += adv_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()

        if epoch > autoencoder_warm_up_n_epochs:
            # Discriminator part
            optimizer_d.zero_grad(set_to_none=True)
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss_d = adv_weight * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

        epoch_loss += recons_loss.item()
        if epoch > autoencoder_warm_up_n_epochs:
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()

        progress_bar.set_postfix(
            {
                "recons_loss": epoch_loss / (step + 1),
                "gen_loss": gen_epoch_loss / (step + 1),
                "disc_loss": disc_epoch_loss / (step + 1),
            }
        )
    epoch_recon_loss_list.append(epoch_loss / (step + 1))
    epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
    epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        autoencoder.eval()
        discriminator.eval()
        model_save_path = os.path.join(save_dir, f'autoenc_epoch_{epoch + 1}.pth')
        torch.save(autoencoder.state_dict(), model_save_path)
        val_batch = first(val_loader)
        images = val_batch["image"].to(device)  # choose only one of Brats channels
        # Generator part
        optimizer_g.zero_grad(set_to_none=True)
        reconstruction, z_mu, z_sigma = autoencoder(images)

        idx = 0
        img = reconstruction[idx, channel].detach().cpu().numpy()
        ori = images[idx, channel].detach().cpu().numpy()

        img_latent = autoencoder.encode_stage_2_inputs(images)[idx, channel].detach().cpu().numpy()

        plt.figure(figsize=(4, 3))
        plt.subplots(3, 3, figsize=(4, 3))
        plt.subplot(3, 3, 1)
        plt.imshow(ori[..., ori.shape[2] // 2], vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.title("ori image")
        plt.subplot(3, 3, 2)
        plt.imshow(ori[:, ori.shape[1] // 2, ...], vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.title("ori image")
        plt.subplot(3, 3, 3)
        plt.imshow(ori[ori.shape[0] // 2, ...], vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.title("ori image")
        plt.subplot(3, 3, 4)
        plt.imshow(img[..., img.shape[2] // 2], vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.title("sampled image")
        plt.subplot(3, 3, 5)
        plt.imshow(img[:, img.shape[1] // 2, ...], vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.title("sampled image")
        plt.subplot(3, 3, 6)
        plt.imshow(img[img.shape[0] // 2, ...], vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.title("sampled image")
        plt.subplot(3, 3, 7)
        plt.imshow(img_latent[..., img_latent.shape[2] // 2], vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.title("sampled image")
        plt.subplot(3, 3, 8)
        plt.imshow(img_latent[:, img_latent.shape[1] // 2, ...], vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.title("sampled image")
        plt.subplot(3, 3, 9)
        plt.imshow(img_latent[img_latent.shape[0] // 2, ...], vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.title("sampled image")
        plt.savefig(os.path.join(save_vis_dir, f'val_vis_E{epoch + 1}.png'))
        plt.close()

del discriminator
del loss_perceptual
torch.cuda.empty_cache()

plt.figure()
plt.style.use("ggplot")
plt.title("Learning Curves", fontsize=20)
plt.plot(epoch_recon_loss_list)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
# plt.show()
plt.savefig(os.path.join(save_vis_dir, f'Learning Curves.png'))
plt.close()

plt.figure()
plt.title("Adversarial Training Curves", fontsize=20)
plt.plot(epoch_gen_loss_list, color="C0", linewidth=2.0, label="Generator")
plt.plot(epoch_disc_loss_list, color="C1", linewidth=2.0, label="Discriminator")
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.savefig(os.path.join(save_vis_dir, f'Adversarial Training Curves.png'))
plt.close()
