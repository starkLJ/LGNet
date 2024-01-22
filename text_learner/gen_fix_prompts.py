import clip
import torch
'''

weather :['daylight','night','foggy']
height   :['low','mid','high']
angle   :[front,side,bird,front-side]
'''
'''
templete = "[weather] view from drone at [height] and [angle] ."
'''

def generate_prompt(weather,height,angle):
    templete = "A [height] altitude [angle] view of a [weather] day taken by a drone."
    prompt = templete.replace("[weather]",weather).replace("[height]",height).replace("[angle]",angle)
    return prompt

def generate_prompt_list():
    weather_list = ['foggy','night','sunny']
    height_list = ['high','medium','low']
    angle_list = ['bird','side','front','front-side']
    prompt_list = []
    for weather in weather_list:
        for height in height_list:
            for angle in angle_list:
                prompt = generate_prompt(weather,height,angle)
                prompt_list.append(prompt)
    return prompt_list

print(generate_prompt_list(),len(generate_prompt_list()))

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

text = generate_prompt_list()
text = clip.tokenize(text).to(device)

model.eval()
text_feats = model.encode_text(text)
print(text_feats.shape)
torch.save(text_feats,'text_learner/text_feats.pt')