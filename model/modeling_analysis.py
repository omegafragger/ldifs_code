import torch
import open_clip
import model.utils as utils

from collections import namedtuple


class ImageEncoderAugmented(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        print(f'Loading {args.model} pre-trained weights.')
        if '__pretrained__' in args.model:
            name, pretrained = args.model.split('__pretrained__')
        else:
            name = args.model
            pretrained = 'openai'
        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=args.openclip_cachedir)
        self.N_slices = 6
        self.conv1 = self.model.visual.conv1
        self.ln_pre = self.model.visual.ln_pre

        self.layers = []
        for i in range(12):
            self.layers.append(self.model.visual.transformer.resblocks[i])

        self.class_embedding = self.model.visual.class_embedding
        self.positional_embedding = self.model.visual.positional_embedding
        self.patch_dropout = self.model.visual.patch_dropout

        self.ln_post = self.model.visual.ln_post
        self._global_pool = self.model.visual._global_pool
        self.proj = self.model.visual.proj

        
        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def get_features(self, images):
        h = self.conv1(images)
        h = h.reshape(h.shape[0], h.shape[1], -1).permute(0, 2, 1)
        h_conv1 = h

        # Class and positional embeddings
        h = torch.cat([self.class_embedding.to(h.dtype) + torch.zeros(h.shape[0], 1, h.shape[-1], dtype=h.dtype, device=h.device), h], dim=1)
        h = h + self.positional_embedding.to(h.dtype)
        h_conv1_with_embedding = h

        h = self.patch_dropout(h)
        h_patch_dropout = h

        h = self.ln_pre(h)
        h_ln_pre = h

        h_layers = []

        h = h.permute(1, 0, 2)
        for r in self.layers:
            h = r(h)
            h_layers.append(h.permute(1, 0, 2))
        h = h.permute(1, 0, 2)

        h = self.ln_post(h)
        h_ln_post = h

        pooled, tokens = self._global_pool(h)
        pooled = pooled @ self.proj
        h_pooled = pooled
        
        outputs = namedtuple("Outputs", ["conv1", "conv1_embedding", "patch_dropout", "ln_pre"] + [f"layer_{i}" for i in range(1, 13)] + ["ln_post", "pooled"])
        out = outputs(h_conv1,
                      h_conv1_with_embedding,
                      h_patch_dropout,
                      h_ln_pre,
                      h_layers[0],
                      h_layers[1],
                      h_layers[2],
                      h_layers[3], 
                      h_layers[4],
                      h_layers[5],
                      h_layers[6],
                      h_layers[7],
                      h_layers[8],
                      h_layers[9],
                      h_layers[10],
                      h_layers[11],
                      h_ln_post,
                      h_pooled)

        return out

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name, filename):
        print(f'Loading image encoder from {filename}')
        state_dict = torch.load(filename)
        return cls.load(model_name, state_dict)

    @classmethod
    def load_from_state_dict(cls, model_name, state_dict):
        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=args.openclip_cachedir)
        self.model.load_from_state_dict(state_dict)