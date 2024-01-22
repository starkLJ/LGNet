import copy
from torch import nn
from torch.nn import functional as F
# Bottelneck of resnet , make_layer
from torchvision.models.resnet import Bottleneck
import torch
from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.detectors.two_stage import TwoStageDetector
import clip
clip_model, preprocess = clip.load("ViT-B/32",device='cpu')
from mmcv.ops import DeformConv2dPack as DCN



@DETECTORS.register_module()
class LGNetSingle(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 custom_cfg = None
                 ):
        super(LGNetSingle, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained, init_cfg)

        # 
        # self.global_encoder = GlobalEncoder(custom_cfg)
        self.global_encoders = nn.ModuleList()
        for i in range(self.neck.num_outs):
            self.global_encoders.append(GlobalEncoder(custom_cfg))
        text_feats_path = custom_cfg['text_feats_path']
        self.text_feats = torch.load(text_feats_path).float().detach()
        

        
        self.sim_loss_weights = custom_cfg['sim_loss_weights']
        self.unsim_loss_weights = custom_cfg['unsim_loss_weights']

    
    

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        super(SingleStageDetector, self).forward_train(img, img_metas)
        if 'attr' in img_metas[0]:
            gt_attrs = [img_meta['attr'] for img_meta in img_metas]
            gt_attrs = torch.Tensor(gt_attrs).long().to(img.device)
        else:
            gt_attrs = None
        features = self.extract_feat(img)
        features = list(features)
        global_feats = []
        sim_loss = None
        for i,feature in enumerate(features):
            
            global_feats.append(self.global_encoders[i](feature))

       
       
        
        unsim_loss = self.get_unsim_loss(global_feats)
        
        sim_loss = self.get_sim_loss(global_feats, img)
        losses = self.bbox_head.forward_train(features, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
    
        if sim_loss is not None:  
            losses.update(sim_loss)
        
        if unsim_loss is not None:
            losses.update(unsim_loss)
        
        return losses

    
    def get_unsim_loss(self, global_feats):
        unsim_loss = torch.tensor(0.0).to(global_feats[0].device)
        
        text_feats = self.text_feats.to(global_feats[0].device).detach()
        text_feats_norm = text_feats / text_feats.norm(dim=-1, keepdim=True)
        for i in range(len(global_feats)):
            global_feats_norm = global_feats[i] / global_feats[i].norm(dim=-1, keepdim=True)
            global_logits = global_feats_norm @ text_feats_norm.t().to(global_feats[0].device)

            tmp_p = (1-global_logits) / 2
            unsim_loss += (-(1-tmp_p)*torch.log(tmp_p)).mean()
        unsim_loss /= len(global_feats)
    
        return {'unsim_loss': unsim_loss * self.unsim_loss_weights}
    
    
    def get_sim_loss(self, global_feats,images):
        image_224 = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            clip_image_feats = clip_model.to(image_224.device).encode_image(image_224)
            clip_image_feats = clip_image_feats / clip_image_feats.norm(dim=-1, keepdim=True)
        
        sim_loss = torch.tensor(0.0).to(image_224.device)
        for i in range(len(global_feats)):
            global_feats_norm = global_feats[i] / global_feats[i].norm(dim=-1, keepdim=True)
            global_logits = global_feats_norm @ clip_image_feats.t()
            tmp_p = (1 + global_logits) / 2
            # get diagonal elements
            tmp_p = torch.diag(tmp_p)
            sim_loss += (-(1-tmp_p) * torch.log(tmp_p)).mean()
            

        sim_loss /= len(global_feats)
    
        return {'sim_loss': sim_loss * self.sim_loss_weights}

    def simple_test(self, img, img_metas, rescale=False):
        features = self.extract_feat(img)

        features = list(features)
        
        results_list = self.bbox_head.simple_test(
            features, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'
        features_list = self.extract_feats(imgs)
        features_list = [list(features) for features in features_list]

        
        
        results_list = self.bbox_head.aug_test(
            features_list, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results


# FSN feature squeeze in paper
class GlobalEncoder(nn.Module):
    '''
    1)down sample the feature map to (b, FPN_outchannels, 16, 16)
    2)encode the feature map to (b, 512, 1, 1)
    '''
    def __init__(self,cfg,size = 16) -> None:
        super().__init__()
        
        self.fpn_outchannels = cfg['fpn_out_channels']
        inchannels = self.fpn_outchannels
        self.dcn_layer = nn.Sequential(
            DCN(inchannels, self.fpn_outchannels, kernel_size=3, stride=1, padding=1, deformable_groups=1),
            nn.InstanceNorm2d(self.fpn_outchannels),
            nn.ReLU(inplace=True),
            DCN(self.fpn_outchannels, self.fpn_outchannels, kernel_size=3, stride=1, padding=1, deformable_groups=1),
            nn.InstanceNorm2d(self.fpn_outchannels),
            nn.ReLU(inplace=True),
        )


        # resnet bolck
        down_sample1 = nn.Sequential(
            nn.Conv2d(self.fpn_outchannels, self.fpn_outchannels, kernel_size=1, stride=2, padding=0),
            nn.InstanceNorm2d(self.fpn_outchannels),
        )
        down_sample2 = nn.Sequential(
            nn.Conv2d(self.fpn_outchannels, self.fpn_outchannels, kernel_size=1, stride=2, padding=0),
            nn.InstanceNorm2d(self.fpn_outchannels),
        )
        down_sample3 = nn.Sequential(
            nn.Conv2d(self.fpn_outchannels, self.fpn_outchannels, kernel_size=1, stride=2, padding=0),
            nn.InstanceNorm2d(self.fpn_outchannels),
        )

        self.encoder = nn.Sequential(
            # nn.AdaptiveMaxPool2d(size),
            nn.AdaptiveAvgPool2d(size),
            Bottleneck(self.fpn_outchannels, self.fpn_outchannels//4, stride=2, downsample=down_sample1,norm_layer=nn.InstanceNorm2d),
            Bottleneck(self.fpn_outchannels, self.fpn_outchannels//4, stride=2, downsample=down_sample2,norm_layer=nn.InstanceNorm2d),
            Bottleneck(self.fpn_outchannels, self.fpn_outchannels//4, stride=2, downsample=down_sample3,norm_layer=nn.InstanceNorm2d),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.fpn_outchannels, 512),
        )

        self.size = size
    
    def _make_layer(self,block,inplanes,planes,stride=1,downsample=None):
        layers = []
        layers.append(block(inplanes,planes,stride,downsample))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.dcn_layer(x)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x


@DETECTORS.register_module()
class LGNet2Stage(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 custom_cfg = None):
        super(LGNet2Stage, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)


        # self.global_encoder = GlobalEncoder(custom_cfg)
        self.global_encoders = nn.ModuleList()
        for i in range(self.neck.num_outs):
            self.global_encoders.append(GlobalEncoder(custom_cfg))
            
        text_feats_path = custom_cfg['text_feats_path']
        self.text_feats = torch.load(text_feats_path).float().detach()

        self.sim_loss_weights = custom_cfg['sim_loss_weights']
        self.unsim_loss_weights = custom_cfg['unsim_loss_weights']

        self.sim_method =  custom_cfg['sim_method']
        
    def forward_train(self,
                    img,
                    img_metas,
                    gt_bboxes,
                    gt_labels,
                    gt_bboxes_ignore=None,
                    gt_masks=None,
                    proposals=None,
                    **kwargs):
        
        x = self.extract_feat(img)
        x = list(x)
        
        ###############################################################
        global_feats = []
        sim_loss = None
        for i, feature in enumerate(x):
            
            global_feats.append(self.global_encoders[i](feature))
        # ################################################
        if self.sim_method == 'l2':
            unsim_loss = self.get_unsim_loss_l2(global_feats)
            sim_loss = self.get_sim_loss_l2(global_feats, img)
        else:
            unsim_loss = self.get_unsim_loss(global_feats)
            sim_loss = self.get_sim_loss(global_feats, img)
       
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        ###############################
        losses.update(unsim_loss)
        losses.update(sim_loss)
        ##################################
        
        return losses
    
    def get_unsim_loss(self, global_feats_after):
        class_ignore = torch.tensor(0.0).to(global_feats_after[0].device)
        text_feats = self.text_feats.to(global_feats_after[0].device).detach()
        text_feats_norm = text_feats / text_feats.norm(dim=-1, keepdim=True)
        for i in range(len(global_feats_after)):
            global_feats_norm = global_feats_after[i] / global_feats_after[i].norm(dim=-1, keepdim=True)
            global_logits = global_feats_norm @ text_feats_norm.t().to(global_feats_after[0].device)

    
            tmp_p = (1-global_logits) / 2
            class_ignore += (-(1-tmp_p)*torch.log(tmp_p)).mean(-1,keepdim=True).mean()
        class_ignore /= len(global_feats_after)
    
        return {'unsim_loss': class_ignore * self.unsim_loss_weights}
    
    def get_unsim_loss_l2(self, global_feats_after):
        class_ignore = torch.tensor(0.0).to(global_feats_after[0].device)
        text_feats = self.text_feats.to(global_feats_after[0].device).detach()
        # text_feats_norm = text_feats / text_feats.norm(dim=-1, keepdim=True)
        for i in range(len(global_feats_after)):
            # global_feats_norm = global_feats_after[i] / global_feats_after[i].norm(dim=-1, keepdim=True)
            # global_logits = global_feats_norm @ text_feats_norm.t().to(global_feats_after[0].device)
            global_feats = global_feats_after[i]
            global_feats = global_feats.unsqueeze(1).repeat(1,text_feats.shape[0],1)
            euclidean_dist = torch.sqrt(torch.sum((global_feats - text_feats.unsqueeze(0)) ** 2, dim=-1))
            tmp_p = 1 - 1 / (1 + euclidean_dist)

            class_ignore += (-(1-tmp_p)*torch.log(tmp_p)).mean(-1,keepdim=True).mean()
        class_ignore /= len(global_feats_after)
    
        return {'unsim_loss': class_ignore * self.unsim_loss_weights}
    
    def get_sim_loss(self, global_feats,images):
        image_224 = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            clip_image_feats = clip_model.to(image_224.device).encode_image(image_224)
            clip_image_feats = clip_image_feats / clip_image_feats.norm(dim=-1, keepdim=True)
        
        sim_loss = torch.tensor(0.0).to(image_224.device)
        for i in range(len(global_feats)):
            global_feats_norm = global_feats[i] / global_feats[i].norm(dim=-1, keepdim=True)
            global_logits = global_feats_norm @ clip_image_feats.t()
            
            tmp_p = (1 + global_logits) / 2
            # get diagonal elements
            tmp_p = torch.diag(tmp_p)
            sim_loss += (-(1-tmp_p) * torch.log(tmp_p)).mean()

           
        sim_loss /= len(global_feats)
    
        return {'sim_loss': sim_loss * self.sim_loss_weights}
    
    def get_sim_loss_l2(self, global_feats,images):
        image_224 = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            clip_image_feats = clip_model.to(image_224.device).encode_image(image_224)
            
        
        sim_loss = torch.tensor(0.0).to(image_224.device)
        for i in range(len(global_feats)):
           
            global_feat = global_feats[i]
            euclidean_dist = torch.sqrt(torch.sum((global_feat - clip_image_feats) ** 2, dim=-1))

            tmp_p = 1 / (1 + euclidean_dist)
           
            sim_loss += (-(1-tmp_p) * torch.log(tmp_p)).mean()

        sim_loss /= len(global_feats)
    
        return {'sim_loss': sim_loss * self.sim_loss_weights}

    
    

