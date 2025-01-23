""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch import nn, Tensor
import os
import warnings
warnings.filterwarnings('ignore')
from model.kanbottleneck import KANBlock,D_ConvLayer,OverlapPatchEmbed
from model.decoderblock import DecoderBottleneck
from model.block import RepLKNetStage,DWLA,ConcatBlock3,ConcatBlock2


class Encoder(nn.Module):
    def __init__(
        self, img_size=200, in_chans=3, embed_dims=[64, 128, 256, 512],
        patch_size = 2,drop_path_rate1=0.3,
        norm_layer=nn.LayerNorm, 
        num_stages=3, pretrained=None,
        drop_rate=0., drop_path_rate=0.,
        large_kernel_sizes=[31,29,27], layers=[2,2,18],
        depths=[1,1,1],

    ):
        super().__init__()
        self.num_stages = num_stages
        self.embed_dims = embed_dims
        self.in_channs = in_chans
        self.patch_size = patch_size
        self.norms_3 = norm_layer(embed_dims[4])
        self.norms_4 = norm_layer(embed_dims[5])

        self.dnorm3 = norm_layer(embed_dims[4])
        self.dnorm4 = norm_layer(embed_dims[3])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate1, sum(layers))]
        dpr1 = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        self.init_weights(pretrained)
        for stage_idx in range(1, self.num_stages+1):
            if stage_idx == 1:
                layer = nn.Sequential(RepLKNetStage(in_channels=in_chans,channels=embed_dims[stage_idx], 
                                                    num_blocks=layers[stage_idx-1],
                                      stage_lk_size=large_kernel_sizes[stage_idx-1],
                                      drop_path=dpr[sum(layers[:stage_idx-1]):sum(layers[:stage_idx])]),
                                     )
            else :
                layer = nn.Sequential(RepLKNetStage(in_channels=embed_dims[stage_idx-1],channels=embed_dims[stage_idx], 
                                      num_blocks=layers[stage_idx-1],
                      stage_lk_size=large_kernel_sizes[stage_idx-1],
                      drop_path=dpr[sum(layers[:stage_idx-1]):sum(layers[:stage_idx])],
                      ),
                                        )
            self.stages.append(layer)
        self.patch_embeds3 = OverlapPatchEmbed(img_size=math.ceil(img_size/(2**3)),
                                            patch_size=3,
                                            stride=2,
                                            in_chans=embed_dims[3],
                                            embed_dim=embed_dims[4])
        self.patch_embeds4 = OverlapPatchEmbed(img_size=img_size//(2**4),
                                    patch_size=3,
                                    stride=2,
                                    in_chans=embed_dims[4],
                                    embed_dim=embed_dims[5])
        self.blocks1 = nn.ModuleList([KANBlock(
            dim=embed_dims[4], 
            drop=drop_rate, drop_path=dpr1[0], norm_layer=norm_layer
            )])

        self.blocks2 = nn.ModuleList([KANBlock(
            dim=embed_dims[5],
            drop=drop_rate, drop_path=dpr1[1], norm_layer=norm_layer
            )])

        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[4], 
            drop=drop_rate, drop_path=dpr1[0], norm_layer=norm_layer
            )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[3], 
            drop=drop_rate, drop_path=dpr1[1], norm_layer=norm_layer
            )])
        self.up1 = nn.Upsample((math.ceil(img_size/16),math.ceil(img_size/16)),mode ='bilinear')
        self.up2 = nn.Upsample((math.ceil(img_size/8),math.ceil(img_size/8)),mode ='bilinear')        
        self.decoder1 = D_ConvLayer(embed_dims[5], embed_dims[4])  
        self.decoder2 = D_ConvLayer(embed_dims[4], embed_dims[3])
        self.decoder3 = D_ConvLayer(embed_dims[3], embed_dims[2])
        self.cat = ConcatBlock2(embed_dims[2])
        self.cat1 = ConcatBlock3(embed_dims[3])
        self.module2_1 = DWLA(embed_dims[1],depths = 1)
        self.module2_2 = DWLA(embed_dims[1],depths = 2)
        self.module3_1 = DWLA(embed_dims[2],depths = 1)
        self.sig = nn.Sigmoid()
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    def forward_features(self, x):
        x1s = x
        outs = []
        for stage_idx in range(self.num_stages):
            x1s = self.stages[stage_idx](x1s)
            outs.append(x1s)
        x2 = outs[0]
        x3 = outs[1]
        x4 = outs[2]
        # SGMAM module
        outs[1] = self.cat(self.module2_1(x2),x3)
        outs[2] = self.cat1(self.module2_2(x2),self.module3_1(x3),x4)
        t3 = outs[-1]
        out, H, W = self.patch_embeds3(x4)
        for i, blk in enumerate(self.blocks1):
            out = blk(out, H, W)
        out = self.norms_3(out)
        B, _, _ = out.shape
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out, H, W= self.patch_embeds4(out)
        for i, blk in enumerate(self.blocks2):
            out = blk(out, H, W)
        out = self.norms_4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(self.up1(self.decoder1(out)))

        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(self.up2(self.decoder2(out)))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.decoder3(out)
        return outs,out

    def forward(self, x):
        outs,out = self.forward_features(x)
        return outs,out


class Decoder(nn.Module):
    def __init__(self, img_dim,out_channels, class_num, decoder_layer=1):
        super().__init__()
        self.decoder1 = DecoderBottleneck(out_channels[2], out_channels[1],img_dim//(2**2),2,m_channels=out_channels[2])
        self.decoder2 = DecoderBottleneck(out_channels[1], out_channels[0],img_dim//(2**1),2,m_channels=out_channels[1])
        self.decoder3 = DecoderBottleneck(out_channels[0], out_channels[0]//2,img_dim//(2**0),2,m_channels=out_channels[0])
        self.img_dim = img_dim
        self.conv1 = nn.Conv2d(out_channels[0]//2, class_num, kernel_size=1)
    def forward(self,outs,x):
        x = self.decoder1(x,outs[-2])
        x = self.decoder2(x,outs[-3])
        x = self.decoder3(x)
        x = self.conv1(x)

        return x


class self_net(nn.Module):
    def __init__(self,  n_channels=1,img_dim=224,embed_dims=[16, 32, 64, 128, 160, 256],filters = [16, 32, 64, 128],layers=[2,2,2],drop_path_rate=0.,drop_path_rate1=0.2,n_classes=2):
        super(self_net, self).__init__()
        self.n_channels=n_channels
        self.n_classes=n_classes
        self.encoder = Encoder(img_dim, n_channels,embed_dims,
                               layers=layers,drop_path_rate1=drop_path_rate1,drop_path_rate=drop_path_rate)

        self.decoder = Decoder(img_dim,filters, n_classes)

    def forward(self, x):

        outs,out = self.encoder(x)
        x = self.decoder(outs,out)

        return x

def calculate_fps(model, input_tensor, num_runs=100):
    # 预热模型，消除启动延迟
    for _ in range(10):
        _ = model(input_tensor)

    # 开始计时
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(input_tensor)
    end_time = time.time()

    # 计算 FPS
    total_time = end_time - start_time
    fps = num_runs / total_time
    return fps
if __name__ == '__main__':
    # import torch
    from thop import profile
    from torchsummary import summary
    import time
    model = self_net(n_channels=1,img_dim=224,n_classes=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 设置为评估模式
    #
    # # 2. 准备输入数据
    #
    # # 模拟单帧输入 (Batch Size = 1)
    # input_data = torch.randn(1, 1, 224, 224).to(device)
    #
    # # 3. 测试推理时间并计算 FPS
    # num_iterations = 100  # 运行多次以计算平均值
    # start_time = time.time()
    #
    # with torch.no_grad():  # 关闭梯度计算以加速推理
    #     for _ in range(num_iterations):
    #         _ = model(input_data)  # 模型推理
    #
    # end_time = time.time()
    #
    # # 计算平均推理时间
    # total_time = end_time - start_time
    # avg_time_per_frame = total_time / num_iterations
    # fps = 1 / avg_time_per_frame
    # summary(model, input_size=(1, 224, 224),batch_size=1)
    #
    # # fps = calculate_fps(model, input_tensor)
    # print(f"FPS: {fps:.2f}")
    # flops, params = profile(model, inputs=(input_data,))
    # print(flops)
    # print(params)
    # print(sum(p.numel() for p in model.parameters()))

    # dummy_input = torch.randn(1, 1, 224, 224, dtype=torch.float).to(device)
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # repetitions = 300
    # timings = np.zeros((repetitions, 1))
    # # GPU-WARM-UP
    # for _ in range(10):
    #     _ = model(dummy_input)
    # # MEASURE PERFORMANCE
    # with torch.no_grad():
    #     for rep in range(repetitions):
    #         starter.record()
    #         _ = model(dummy_input)
    #         ender.record()
    #         # WAIT FOR GPU SYNC
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)
    #         timings[rep] = curr_time
    # mean_syn = np.sum(timings) / repetitions
    # std_syn = np.std(timings)
    # mean_fps = 1000. / mean_syn
    # print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn,
    #                                                                                      std_syn=std_syn,
    #                                                                                      mean_fps=mean_fps))
