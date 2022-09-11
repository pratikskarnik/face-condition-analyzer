import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('export.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Face condition Analyzer"
description = "A face condition detector trained on the custom dataset with fastai. Created using Gradio and HuggingFace Spaces."
examples = ['harmonal_acne.jpg','forehead_wrinkles.jpg','oily_skin.jpg']
enable_queue=True

gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=3),title=title,
             description=description,examples=examples,enable_queue=enable_queue).launch(share=True)