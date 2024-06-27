import torch
import open_clip
import model.utils as utils

from collections import namedtuple


class ImageEncoder(torch.nn.Module):
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
        
        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')


    def forward(self, images, text=None):
        # This provides the following output:
        # Tuple: (image_feature, text_feature, logit_scale.exp())
        assert self.model is not None
        if text == None:
            return self.model.encode_image(images)
        if images == None:
            return self.model.encode_text(text)
        return self.model(images, text)


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
        self.N_slices = 5
        self.conv1 = self.model.visual.conv1
        self.ln_pre = self.model.visual.ln_pre
        self.layer1 = self.model.visual.transformer.resblocks[0:3]
        self.layer2 = self.model.visual.transformer.resblocks[3:6]
        self.layer3 = self.model.visual.transformer.resblocks[6:9]
        self.layer4 = self.model.visual.transformer.resblocks[9:12]

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

        # Class and positional embeddings
        h = torch.cat([self.class_embedding.to(h.dtype) + torch.zeros(h.shape[0], 1, h.shape[-1], dtype=h.dtype, device=h.device), h], dim=1)
        h = h + self.positional_embedding.to(h.dtype)
        h = self.patch_dropout(h)
        h = self.ln_pre(h)
        h_ln_pre = h

        h = h.permute(1, 0, 2)
        for r in self.layer1:
            h = r(h)
        h_layer1 = h.permute(1, 0, 2)
        for r in self.layer2:
            h = r(h)
        h_layer2 = h.permute(1, 0, 2)
        for r in self.layer3:
            h = r(h)
        h_layer3 = h.permute(1, 0, 2)
        for r in self.layer4:
            h = r(h)
        h = h.permute(1, 0, 2)
        h_layer4 = h

        h = self.ln_post(h)
        h_ln_post = h

        pooled, tokens = self._global_pool(h)
        pooled = pooled @ self.proj

        
        outputs = namedtuple("Outputs", ["lnpre", "layer1", "layer2", "layer3", "layer4"])
        out = outputs(h_ln_pre, h_layer1, h_layer2, h_layer3, h_layer4)

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


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)