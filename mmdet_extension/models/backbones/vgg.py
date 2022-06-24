from mmcv.cnn import VGG as VGGCV

from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class VGG(VGGCV):
    def __init__(self,
                 depth,
                 *args,
                 **kwargs
                 ):
        super().__init__(depth=depth, *args, **kwargs)

    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)

    def forward(self, x):
        outs = []
        vgg_layers = getattr(self, self.module_name)
        for i in range(len(self.stage_blocks)):
            for j in range(*self.range_sub_modules[i]):
                vgg_layer = vgg_layers[j]
                x = vgg_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
