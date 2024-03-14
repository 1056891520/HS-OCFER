import torch
import torch.nn as nn
import torchvision
DIM = 256

class MyConvo2d(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True, stride=1, bias=True):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1) / 2)
        self.conv = torch.nn.Conv2d(input_dim, output_dim, kernel_size, stride=stride, padding=self.padding, bias=bias)

    def forward(self, x):
        output = self.conv(x)
        return output

class MeanPoolConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init=self.he_init)

    def forward(self, x):
        output = x
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output[:, :, ::2, 1::2]
                  + output[:, :, 1::2, 1::2]) / 4
        output = self.conv(output)
        return output

class ConvMeanPool(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init=self.he_init)

    def forward(self, x):
        output = self.conv(x)
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output[:, :, ::2, 1::2] +
                  output[:, :, 1::2, 1::2]) / 4
        return output

class DepthToSpace(torch.nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, x):
        output = x.permute(0, 2, 3, 1)
        output = output.contiguous()
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size, input_height, output_width, output_depth) for t_t in spl]
        output = torch.stack(stacks, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, output_height,
                                                                                      output_width, output_depth)
        output = output.contiguous()
        output = output.permute(0, 3, 1, 2)
        output = output.contiguous()
        return output

class UpSampleConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True, bias=True):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init=self.he_init, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, x):
        output = x
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output

class ResidualBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=DIM, encoder=False):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        if resample == 'down':
            if encoder is True:
                self.bn1 = torch.nn.BatchNorm2d(input_dim)
                self.bn2 = torch.nn.BatchNorm2d(input_dim)
            else:
                self.bn1 = torch.nn.LayerNorm([input_dim, hw, hw])
                self.bn2 = torch.nn.LayerNorm([input_dim, hw, hw])
        elif resample == 'up':
            self.bn1 = torch.nn.BatchNorm2d(input_dim)
            self.bn2 = torch.nn.BatchNorm2d(output_dim)
        elif resample is None:
            self.bn1 = torch.nn.BatchNorm2d(output_dim)
            self.bn2 = torch.nn.LayerNorm([input_dim, hw, hw])
        else:
            raise Exception('invalid resample value')

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size=1, he_init=False)
            self.conv_1        = MyConvo2d(input_dim, input_dim, kernel_size=kernel_size, bias=False)
            self.conv_2        = ConvMeanPool(input_dim, output_dim, kernel_size=kernel_size)
        elif resample == 'up':
            self.conv_shortcut = UpSampleConv(input_dim, output_dim, kernel_size=1, he_init=False)
            self.conv_1        = UpSampleConv(input_dim, output_dim, kernel_size=kernel_size, bias=False)
            self.conv_2        = MyConvo2d(output_dim, output_dim, kernel_size=kernel_size)
        elif resample is None:
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size=1, he_init=False)
            self.conv_1        = MyConvo2d(input_dim, input_dim, kernel_size=kernel_size, bias=False)
            self.conv_2        = MyConvo2d(input_dim, output_dim, kernel_size=kernel_size)
        else:
            raise Exception('invalid resample value')

    def forward(self, x):
        if self.input_dim == self.output_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.conv_shortcut(x)

        output = x
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output

class Encoder_224(torch.nn.Module):
    def __init__(self, z_dim=128):
        super(Encoder_224, self).__init__()
        self.dim = 64
        self.conv1 = MyConvo2d(3, self.dim, kernel_size=5, stride=2, he_init=False)    # 3
        self.rb1 = ResidualBlock(1 * self.dim, 2 * self.dim, 3, resample='down', hw=int(DIM / 1), encoder=True)
        self.rb2 = ResidualBlock(2 * self.dim, 4 * self.dim, 3, resample='down', hw=int(DIM / 2), encoder=True)
        self.rb3 = ResidualBlock(4 * self.dim, 8 * self.dim, 3, resample='down', hw=int(DIM / 4), encoder=True)
        self.rb4 = ResidualBlock(8 * self.dim, 8 * self.dim, 3, resample='down', hw=int(DIM / 8), encoder=True)
        self.conv2 = MyConvo2d(8 * self.dim, 8 * self.dim, kernel_size=3, stride=2, he_init=False)
        self.ln1 = nn.Linear(7 * 7 * 8 * self.dim, z_dim)

#
    def forward(self, x):
        output = x.contiguous()
        output = output.view(-1, 3, 224, 224)   # (bs, 3, 224, 224)
        conv1_output = self.conv1(output)     # (bs, 64, 112, 112)
        rb1_output = self.rb1(conv1_output)     # (bs, 128, 56, 56)
        rb2_output = self.rb2(rb1_output)     # (bs, 256, 28, 28)
        rb3_output = self.rb3(rb2_output)     # (bs, 512, 14, 14)
        rb4_output = self.rb4(rb3_output)     # (bs, 512, 7, 7)
        # output = self.conv2(output)
        z = rb4_output.view(-1, 7 * 7 * 8 * self.dim)  # (bs, 25088)
        z = self.ln1(z)  # (bs, 128)
        return z, rb4_output, rb3_output, rb2_output, rb1_output, conv1_output

class Decoder_224(torch.nn.Module):
    def __init__(self, z_dim=128, ):
        super(Decoder_224, self).__init__()
        self.dim = 64
        self.ln1 = torch.nn.Linear(z_dim, 7 * 7 * (8 * self.dim))
        self.up1 = UpSampleConv(8 * self.dim, 8 * self.dim, kernel_size=3)
        self.rb1 = ResidualBlock(8 * self.dim, 8 * self.dim, 3, resample='up')
        self.rb2 = ResidualBlock(8 * self.dim, 4 * self.dim, 3, resample='up')
        self.rb3 = ResidualBlock(4 * self.dim, 2 * self.dim, 3, resample='up')
        self.rb4 = ResidualBlock(2 * self.dim, 1 * self.dim, 3, resample='up')
        self.bn = torch.nn.BatchNorm2d(self.dim)
        self.conv1 = MyConvo2d(1 * self.dim, 3, 3)
        self.relu = torch.nn.ReLU()
        self.up2 = UpSampleConv(self.dim, 3, kernel_size=5)

    def forward(self, x, bn1_rec, bn2_rec, bn3_rec):
        output = self.ln1(x)    # (bs, 25088)
        output = output.view(-1, 8 * self.dim, 7, 7)     # (bs, 512, 7, 7)

        up1_output = self.up1(output)     # (bs, 512, 14, 14)
        rb1_output = self.rb1(up1_output)     # (bs, 512, 28, 28)
        rb2_output = self.rb2(rb1_output)+bn3_rec     # (bs, 256, 56, 56)
        rb3_output = self.rb3(rb2_output)+bn2_rec     # (bs, 128, 112, 112)
        rb4_output = self.rb4(rb3_output)+bn1_rec     # (bs, 64, 224, 224)
        bn_output = self.bn(rb4_output)
        y = self.relu(bn_output)
        # output = self.up2(output)
        y = self.conv1(y)     # (bs, 3, 224, 224)
        # return y, bn_output, rb4_output, rb3_output, rb2_output, rb1_output, up1_output
        return y

def HFE_Module(x):
    out = nn.Conv2d(x.shape[1], x.shape[1] // 2, 3, 1, 1, bias=True)(x)
    out = nn.BatchNorm2d(x.shape[1] // 2)(out)
    out_latent = nn.ReLU()(out)

    z_latent = nn.Conv2d(x.shape[1] // 2, x.shape[1] // 2, 3, 1, 1, bias=True)(out_latent)
    z_latent = nn.MaxPool2d(z_latent.shape[2], z_latent.shape[2])(z_latent)
    z_latent = z_latent.view(-1, z_latent.shape[1])

    rec_latent = UpSampleConv(out_latent.shape[1], out_latent.shape[1], kernel_size=3)(out_latent)
    rec_latent = UpSampleConv(rec_latent.shape[1], rec_latent.shape[1], kernel_size=3)(rec_latent)

    return z_latent, rec_latent


class HS_Model(nn.Module):
    def __init__(self):
        super(HS_Model, self).__init__()
        self.encoder = Encoder_224()
        self.decoder = Decoder_224()
        self.fc = nn.Linear(64+128+256+256, 24)
        self.ou_loss = nn.MSELoss()

    def forward(self, x):
        encoder_z, encoder_rb4, encoder_rb3, encoder_rb2, encoder_rb1, encoder_conv1 = self.encoder(x)
        bn1_z, bn1_rec = HFE_Module(encoder_rb1)  # (1,64); (1,64,224,224)
        bn2_z, bn2_rec = HFE_Module(encoder_rb2)  # (1,128); (1,128,112,112)
        bn3_z, bn3_rec = HFE_Module(encoder_rb3)  # (1,256); (1,256,56,56)
        xrec= self.decoder(encoder_z, bn1_rec, bn2_rec, bn3_rec)
        z_latent = torch.cat((encoder_z, bn1_z, bn2_z, bn3_z), dim=1)
        landmark_pred = self.fc()(z_latent)

        return xrec, landmark_pred, z_latent

    def loss(self, x, xrec, landmark, landmark_pred,):
        xrec_loss = self.ou_loss(x, xrec)
        landmark_loss = self.ou_loss(landmark, landmark_pred)

        return xrec_loss, landmark_loss

# if __name__ == '__main__':
#     encoder = Encoder_224()
#     decoder = Decoder_224()
#     x = torch.rand((1, 224, 224, 3))
#     encoder_z, encoder_rb4, encoder_rb3, encoder_rb2, encoder_rb1, encoder_conv1 = encoder(x)
#     bn1_z, bn1_rec = HFE_Module(encoder_rb1)    # (1,64); (1,64,224,224)
#     bn2_z, bn2_rec = HFE_Module(encoder_rb2)    # (1,128); (1,128,112,112)
#     bn3_z, bn3_rec = HFE_Module(encoder_rb3)    # (1,256); (1,256,56,56)
#     # z_latent, rec_latent = HFE_Module(encoder_rb1)
#     decoder_y, decoder_bn, decoder_rb4, decoder_rb3, decoder_rb2, decoder_rb1, decoder_up1 = decoder(encoder_z, bn1_rec, bn2_rec, bn3_rec)
#
#
#     print(y.shape)
