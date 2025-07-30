"""Entry to launch gradio app."""
import gradio as gr
from src.demo.gradio_utils import *


with gr.Blocks(css="style.css") as demo:
    gr.Markdown("# 🎵 AudioMorphix Gradio Demo 🎵\n<p>Select a task and edit audio interactively.</p>")
    with gr.Tabs():
        with gr.TabItem("Mix Audio"):
            create_add_demo()
        with gr.TabItem("Remove Audio"):
            create_remove_demo()
        with gr.TabItem("Move & Resize Audio"):
            create_move_demo()

demo.queue(max_size=20)
demo.launch(
    debug=True, 
    server_name="0.0.0.0", 
    server_port=7200,
    )  # comment the url if using space