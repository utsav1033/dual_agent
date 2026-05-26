import gradio as gr

def respond(msg, history):
    history = history or []
    history.append({"role": "user", "content": msg})
    history.append({"role": "assistant", "content": f"You said: {msg}"})
    return history, ""

with gr.Blocks() as demo:
    bot = gr.Chatbot(label="Test", height=300)
    inp = gr.Textbox(placeholder="Type here…")
    btn = gr.Button("Send")
    btn.click(respond, [inp, bot], [bot, inp])

demo.launch(server_port=7861, prevent_thread_lock=True)

import time, urllib.request
time.sleep(3)
r = urllib.request.urlopen("http://127.0.0.1:7861")
print("HTTP", r.status)
