import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from minestudio.data import RawDataModule
from minestudio.data.minecraft.callbacks import ImageKernelCallback, ActionKernelCallback, SegmentationKernelCallback
from minestudio.offline import MineLightning
from minestudio.models import VPTPolicy
from minestudio.offline.mine_callbacks import BehaviorCloneCallback
from minestudio.offline.lightning_callbacks import SmartCheckpointCallback, SpeedMonitorCallback

policy = VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.foundation_model_2x")
mine_lightning = MineLightning(
    mine_policy=policy,
    learning_rate=0.00004,
    warmup_steps=2000,
    weight_decay=0.000181,
    callbacks=[BehaviorCloneCallback(weight=0.01)]
)

episode_continuous_batch = True
mine_data = RawDataModule(
    data_params=dict(
        dataset_dirs=['10xx'],
        modal_kernel_callbacks=[
            ImageKernelCallback(frame_width=128, frame_height=128, enable_video_aug=False),
            ActionKernelCallback(enable_prev_action=True, win_bias=1, read_bias=-1),
        ],
        win_len=128,
        split_ratio=0.9,
        shuffle_episodes=True,
    ),
    batch_size=1,
    num_workers=1,
    prefetch_factor=1,
    episode_continuous_batch=episode_continuous_batch,
)
mine_data.setup("fit")
train_loader = mine_data.train_dataloader()   # это DataLoader
batch = next(iter(train_loader))              # берём первый батч

print(type(batch))
print(batch.keys())
for k, v in batch.items():
    print(k, getattr(v, "shape", type(v)))

L.Trainer(
    logger=WandbLogger(project="minestudio-vpt"),
    devices=1,
    strategy='auto',
    precision="bf16",
    use_distributed_sampler=False,
    gradient_clip_val=1.0,
    callbacks=[
        LearningRateMonitor(logging_interval='step'),
        SpeedMonitorCallback(),
        SmartCheckpointCallback(
            dirpath='./weights',
            filename='weight-{epoch}-{step}',
            save_top_k=-1,
            every_n_train_steps=2000,
            save_weights_only=True,
        ),
        SmartCheckpointCallback(
            dirpath='./checkpoints',
            filename='ckpt-{epoch}-{step}',
            save_top_k=1,
            every_n_train_steps=2001,
            save_weights_only=False,
        )
    ]
).fit(model=mine_lightning, datamodule=mine_data)
