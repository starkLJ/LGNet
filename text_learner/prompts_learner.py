import torch
from torch import nn
import json
import clip
from PIL import Image
from torch.nn import functional as F

from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import os
import cv2

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self,clip_model,prompt_template:list,learning_pos:list,learning_counts:list) -> None:
        
        super().__init__()

        prompt = clip.tokenize(prompt_template).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt)

        hight_prompt = ['high','medium','low']
        angle_prompt = ['bird','side','front','other']

        hight_prompt = clip.tokenize(hight_prompt).cuda()
        angle_prompt = clip.tokenize(angle_prompt).cuda()
        with torch.no_grad():
            hight_embedding = clip_model.token_embedding(hight_prompt)
            angle_embedding = clip_model.token_embedding(angle_prompt)

        # shape = [n_learning_vectors, embedding_dim]
        learn_vec_list = nn.ParameterList()
        self.counts = 1
        for idx , counts in enumerate(learning_counts):
            # learn_vec_list.append(nn.Parameter(learning_vectors[idx].repeat(counts,1)))
            if idx == 0:
                learn_vec_list.append(nn.Parameter(hight_embedding[:,1]))
            else:
                learn_vec_list.append(nn.Parameter(angle_embedding[:,1]))
            self.counts *= counts
        self.learn_vec_list = learn_vec_list

        self.pos = learning_pos
        

        self.freeze_embedding = embedding

        self.tokenized_prompts = prompt[0]

    def forward(self):
        # shape = [batch_size, n_ctx, embedding_dim]
        
        embedding = self.freeze_embedding.unsqueeze(1).repeat(1,self.counts,1,1).reshape(36,-1,512)
        height_vec = self.learn_vec_list[0].reshape(3,1,-1).repeat(1,4,1).reshape(12,1,-1).repeat(3,1,1).reshape(36,-1)
        angle_vec  = self.learn_vec_list[1].reshape(4,1,-1).repeat(3,1,1).reshape(12,1,-1).repeat(3,1,1).reshape(36,-1)
        
        embedding[:,self.pos[0],:] = height_vec
        embedding[:,self.pos[1],:] = angle_vec
        return embedding

class CustomCLIP(nn.Module):
    def __init__(self,clip_model) -> None:
        super().__init__()
        prompt_list = [
             "A alpha altitude beta view of a foggy day taken by a drone.",
             "A alpha altitude beta view of a night day taken by a drone.",
             "A alpha altitude beta view of a sunny day taken by a drone.",

            # "foggy view from drone at alpha altitude and beta angle .",
            # "night view from drone at alpha altitude and beta angle .",
            # "daylight view from drone at alpha altitude and beta angle ."
        ]

        learning_pos = [2,4]
        learning_counts = [3,4]
        self.prompt_learner = PromptLearner(clip_model,prompt_list,learning_pos,learning_counts)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner().type(self.dtype)
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits




class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root,json_file, preprocess) -> None:
        super().__init__()
        self.data = json.load(open(json_file))['images']
        self.preprocess = preprocess
        # sample interval 30
        self.data = self.data[::10]
        self.root = data_root
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data[idx]
        file_name = data_dict["file_name"]
        
        file_name = os.path.join(self.root,file_name)
        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.preprocess(image = image)["image"]

        attr = data_dict["attr"]
        '''

        label = weather * 12 + attitude * 4 + view
        '''

        

        label = attr

        return image, label




def main():
    

    model, clip_preprocess = clip.load("ViT-B/32", device="cuda:0")

   
    preprocess = A.Compose(
        [
        A.HorizontalFlip(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.OneOf(
            [
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            ]
            ,
            p=0.1,
        ),
        A.JpegCompression(quality_lower=85, quality_upper=95, p=0.2),
        A.OneOf(
            [
                A.Blur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ],
            p=0.1,
        ),
        A.Resize(224, 224),
        A.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ToTensorV2()
        ]
    )
    custom_clip = CustomCLIP(model)
    custom_clip = custom_clip.cuda()
    for name,param in custom_clip.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
    optimizer = torch.optim.SGD(custom_clip.prompt_learner.parameters(),lr=0.002,momentum=0.9)

    train_dataset = Dataset('DATA/UAVDT/images/train','DATA/UAVDT/annotations/UAVDT_train_coco.json',preprocess)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=512,shuffle=True)
        
    EPOCHS = 15
    for epoch in range(EPOCHS):
        custom_clip.train()

        avg_train_loss = 0
        
        for idx,(image,label) in enumerate(train_loader):
            image = image.cuda()
            label = label.cuda()
            logits = custom_clip(image)
            loss = F.cross_entropy(logits,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_train_loss += loss.item()
            
            sys.stdout.write(f"\repoch {epoch} batch {idx} loss {avg_train_loss/(idx+1)}")
            sys.stdout.flush()
        print()

        
        # save model

        torch.save(custom_clip.prompt_learner.state_dict(),f"text_learner/prompt_learner_last.pth")

if __name__ == "__main__":
    
    main()


    # ''' register forward hook CustomCLIP forward text_features '''
    output_text_feats = []
    def hook(module, input, output):
        output_text_feats.append(output)
    
    model, preprocess = clip.load("ViT-B/32", device="cuda:0")
    custom_clip = CustomCLIP(model)
    # if have checkpoints
    # custom_clip.prompt_learner.load_state_dict(torch.load("models/clip_29.pth"))
    custom_clip.text_encoder.register_forward_hook(hook)
    custom_clip.eval()
    image = torch.randn(1,3,224,224).cuda()
    logits = custom_clip(image)
    
    save_text_feats = output_text_feats[0]
    fix_text_feats = torch.load("text_learner/text_feats.pt")
    save_text_feats = (save_text_feats + fix_text_feats) / 2
    torch.save(save_text_feats,"text_learner/uavdt_text_feats.pth")

    