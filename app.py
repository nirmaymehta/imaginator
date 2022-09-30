from cgitb import text
import tkinter as tk
from tkinter.ttk import Label
import customtkinter as ctk
from authtoken import auth_token
from tkinter import * 
from tkinter.ttk import *

from PIL import ImageTk

import torch
import os
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(height=40, width=512, text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(height=512, width=512)
lmain.place(x=10, y=110)

lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

modelid = "CompVis/stable-diffusion-v1-4"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipe = StableDiffusionPipeline.from_pretrained(
    modelid, 
    scheduler=lms,
    use_auth_token=auth_token,
    cache_dir=os.getenv("cache_dir", "./models")
).to(device)

def generate():
    if device == "cuda":
        with autocast(device):
            image = pipe(
                prompt.get(),
                guidance_scale=5,
                )["sample"][0]
    else:
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]

    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(file='generatedimage.png')
    imgLabel = tk.Label(image=img)
    imgLabel.pack()
    
    

trigger = ctk.CTkButton(height=40, width=512, text_color="white", fg_color="green", text="Generate an image!", command=generate)
trigger.place(x=10, y=60)

app.mainloop()