import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2))

        # linear layers
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=10)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def sub_forward(self, x):
        """
        Forward pass the input image through 1 subnetwork
        :param x: Tensor of size (B,C,H,W)
        :return out: Tensor of size (B, 2)
        """
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))

        out = out.view(-1, 256 * 6 * 6)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

    def forward(self, x1, x2):
        """
        Forward pass input image pairs through both subtwins and return the encoded versions of the image
        :param x1: a Tensor of size (B, C, H, W). The left image pairs along the batch dimension
        :param x2: a Tensor of size (B, C, H, W). The right image pairs along the batch dimension.
        :return encoded images: a Tensor of size (B, 2).
        """

        # encode image pairs
        h1 = self.sub_forward(x1)
        h2 = self.sub_forward(x2)

        return h1, h2


class G_NET(nn.Module):
    def __init__(self, gf_dim, branch_num, ef_dim, t_dim, z_dim):
        super(G_NET, self).__init__()
        self.ca_net = CA_NET(ef_dim=ef_dim, t_dim=t_dim)
        self.gf_dim = gf_dim
        self.z_dim = z_dim
        self.branch_num = branch_num
        self.ef_dim = ef_dim
        self.define_module()

    def define_module(self):
        if self.branch_num > 0:
            self.h_net1 = INIT_STAGE_G(self.gf_dim * 16, z_dim=self.z_dim, embedding_dim=self.ef_dim)
            self.img_net1 = GET_IMAGE_G(self.gf_dim)
        if self.branch_num > 1:
            self.h_net2 = NEXT_STAGE_G(self.gf_dim, embedding_dim=self.ef_dim)
            self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)
        if self.branch_num > 2:
            self.h_net3 = NEXT_STAGE_G(self.gf_dim // 2, embedding_dim=self.ef_dim)
            self.img_net3 = GET_IMAGE_G(self.gf_dim // 4)
        if self.branch_num > 3:  # Recommended structure (mainly limited by GPU memory), and not test yet
            self.h_net4 = NEXT_STAGE_G(self.gf_dim // 4, num_residual=1, embedding_dim=self.ef_dim)
            self.img_net4 = GET_IMAGE_G(self.gf_dim // 8)
        if self.branch_num > 4:
            self.h_net4 = NEXT_STAGE_G(self.gf_dim // 8, num_residual=1, embedding_dim=self.ef_dim)
            self.img_net4 = GET_IMAGE_G(self.gf_dim // 16)

    def forward(self, z_code, text_embedding=None):

        c_code, mu, logvar = self.ca_net(text_embedding)
        fake_imgs = []
        if self.branch_num > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if self.branch_num > 1:
            h_code2 = self.h_net2(h_code1, c_code)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
        if self.branch_num > 2:
            h_code3 = self.h_net3(h_code2, c_code)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
        if self.branch_num > 3:
            h_code4 = self.h_net4(h_code3, c_code)
            fake_img4 = self.img_net4(h_code4)
            fake_imgs.append(fake_img4)

        return fake_imgs, mu, logvar


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self, t_dim, ef_dim):
        super(CA_NET, self).__init__()
        self.t_dim = t_dim
        self.ef_dim = ef_dim
        self.fc = nn.Linear(self.t_dim, self.ef_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.ef_dim]
        logvar = x[:, self.ef_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, z_dim, embedding_dim):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = z_dim + embedding_dim
        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())

        self.upsample1 = up_block(ngf, ngf // 2)
        self.upsample2 = up_block(ngf // 2, ngf // 4)
        self.upsample3 = up_block(ngf // 4, ngf // 8)
        self.upsample4 = up_block(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code=None):
        in_code = torch.cat((c_code, z_code), 1)
        # state size 16ngf x 4 x 4
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size 8ngf x 8 x 8
        out_code = self.upsample1(out_code)
        # state size 4ngf x 16 x 16
        out_code = self.upsample2(out_code)
        # state size 2ngf x 32 x 32
        out_code = self.upsample3(out_code)
        # state size ngf x 64 x 64
        out_code = self.upsample4(out_code)

        return out_code


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, embedding_dim, num_residual=2):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = embedding_dim
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim

        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.upsample = up_block(ngf, ngf // 2)

    def forward(self, h_code, c_code):
        s_size = h_code.size(2)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)
        # state size (ngf+egf) x in_size x in_size
        h_c_code = torch.cat((c_code, h_code), 1)
        # state size ngf x in_size x in_size
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code


def up_block(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


def conv3x3(in_planes, out_planes):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(1, 1),
                     padding=1, bias=False)


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class SiameseResNetwork(ResNet):
    def __init__(self):
        super(SiameseResNetwork, self).__init__(BasicBlock, [2,2,2,2])

    def twinforward(self, x1, x2):
        output1 = super().forward(x1)
        output2 = super().forward(x2)

        return output1, output2

