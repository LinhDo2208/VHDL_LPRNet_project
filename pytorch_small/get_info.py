from torchinfo import summary
from LPRNet import build_lprnet

model = build_lprnet(lpr_max_len=8, phase="train", class_num=36, dropout_rate=0.5)

summary(model=model,
        input_size=(1, 3, 24, 94),  # (batch_size, color_channels, height, width)
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])