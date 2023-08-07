import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.base_cmn import BaseCMNITHN
from modules.visual_extractor import VisualExtractor
    
class ImgProjection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x: torch.Tensor):
        img_feats = self.linear1(x)
        img_feats = img_feats / img_feats.norm(dim = 1, keepdim = True)

        return img_feats

class TextProjection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x: torch.Tensor):
        text_feats = self.linear1(x)
        text_feats = text_feats / text_feats.norm(dim = 1, keepdim = True)

        return text_feats

class BaseCMNModelITHN(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseCMNModelITHN, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = BaseCMNITHN(args, tokenizer)
        self.forward = self.forward_cxr
        self.img_projection = ImgProjection(2048, 512)
        self.text_projection = TextProjection(512, 512)
        self.temp_1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.temp_3 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_cxr(self, images, neg_images = None, targets = None, targets_mask = None, neg_targets = None, neg_targets_mask = None, lambd = 1.0, mode='train', update_opts={}):
        att_feats, fc_feats = self.visual_extractor(images)
        if neg_images is not None:
            neg_att_feats, neg_fc_feats = self.visual_extractor(neg_images)
        if mode == 'train':
            pos_rep, pos_rep_output = self.encoder_decoder(fc_feats, att_feats, targets, neg_seq = True, mode='forward')
            _, neg_rep_output = self.encoder_decoder(neg_fc_feats, neg_att_feats, neg_targets, neg_seq = True, mode='forward')

            pos_img_embeds = self.img_projection(fc_feats)
            neg_img_embeds = self.img_projection(neg_fc_feats)

            pos_rep_embeds = self.text_projection(torch.mean(pos_rep_output, dim = 1))
            neg_rep_embeds = self.text_projection(torch.mean(neg_rep_output, dim = 1))

            zi_uni = torch.sub(pos_img_embeds, neg_rep_embeds)
            ui_uni = torch.sub(pos_rep_embeds, neg_rep_embeds)

            p = neg_rep_embeds + torch.matmul(zi_uni @ ui_uni.t(), F.normalize(ui_uni, dim = -1))
            h_neg_embeds = lambd * neg_rep_embeds + (1 - lambd) * p

            similarity_matrix = self.temp_1.exp() * pos_img_embeds @ pos_rep_embeds.t()
            n_similarity_matrix = self.temp_1.exp() * neg_img_embeds @ neg_rep_embeds.t()
            h_n_similarity_matrix = self.temp_3.exp() * pos_img_embeds @ h_neg_embeds.t()
            h_n_similarity_matrix = h_n_similarity_matrix * torch.diag(torch.ones(h_n_similarity_matrix.shape[0], device=h_n_similarity_matrix.device) * -1)
            fin_similarity_matrix = (similarity_matrix + n_similarity_matrix + h_n_similarity_matrix ) / 3

            l_cp = self.clip_loss(fin_similarity_matrix)
            l_cs = F.cosine_similarity(pos_img_embeds, neg_img_embeds).abs().mean()

            return pos_rep, l_cp, l_cs
        
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError

    def contrastive_loss(self, logits: torch.Tensor):
        return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device = logits.device))
    
    def clip_loss(self, similarity: torch.Tensor):
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0