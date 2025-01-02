import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

def get_vit_cifar100():
    """
    Returns a pre-trained ViT-16 model configured for CIFAR-100.
    The model is based on the 'vit_base_patch16_224' architecture.
    
    Returns:
        model (nn.Module): Pre-trained ViT model
    """
    # Initialize the ViT model with original settings
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=100)
    
    # Add a hook to get the embeddings from the last layer
    def get_embeddings(self, x):
        # Forward pass through all layers except the head
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        # Return the CLS token embeddings
        return x[:, 0]
    
    # Add the method as an attribute
    model.get_embeddings = get_embeddings.__get__(model)

    return model


class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.vit = get_vit_cifar100()

    def get_embeddings(self, x):
        return self.vit.get_embeddings(x)

    def forward(self, x):
        embed = self.get_embeddings(x)
        # pred = F.normalize(embed, dim=1)
        pred = embed
        out = self.vit.head(pred)

        return pred, out


        