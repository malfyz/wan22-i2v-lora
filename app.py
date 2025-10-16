import gradio as gr
import subprocess, threading, os, shlex, time, glob

WORKDIR = "/workspace"
DATASETS_DIR = f"{WORKDIR}/datasets"
CHAR_DIR = f"{DATASETS_DIR}/character_images"
LOG_DIR = f"{WORKDIR}/logs"
SCRIPTS = {
    "Train HN": f"{WORKDIR}/scripts/train_i2v_high.sh",
    "Train LN": f"{WORKDIR}/scripts/train_i2v_low.sh",
}
os.makedirs(LOG_DIR, exist_ok=True)

PROCS = {}
LOGFILES = {k: f"{LOG_DIR}/{k.replace(' ', '_').lower()}.log" for k in SCRIPTS}

def _run_bg(name, cmd):
    with open(LOGFILES[name], "w", buffering=1) as lf:
        lf.write(f"$ {cmd}\n")
        lf.flush()
        proc = subprocess.Popen(["/bin/bash", "-lc", cmd], stdout=lf, stderr=lf, text=True)
        PROCS[name] = proc
        proc.wait()

def download_dataset(url):
    url = (url or "").strip()
    if not url.startswith("http"):
        return f"Invalid URL: {url}"
    os.makedirs(CHAR_DIR, exist_ok=True)
    cmd = f'cd {DATASETS_DIR} && aria2c -x 8 -s 8 "{url}" -o dataset_archive && ' \
          f'(file -b dataset_archive | grep -qi zip && unzip -o dataset_archive -d character_images || ' \
          f'(mkdir -p tmp && tar -xf dataset_archive -C tmp && rsync -a tmp/ character_images/ && rm -rf tmp)) && ' \
          f'rm -f dataset_archive && ' \
          f'(shopt -s nullglob && if compgen -G "{CHAR_DIR}/*/*" > /dev/null; then mv {CHAR_DIR}/*/* {CHAR_DIR}/ && ' \
          f'find {CHAR_DIR} -type d -empty -delete; fi)'
    out = subprocess.run(["/bin/bash", "-lc", cmd], capture_output=True, text=True)
    if out.returncode != 0:
        return f"Download/Extract failed:\n{out.stderr or out.stdout}"
    count = len([p for p in glob.glob(f"{CHAR_DIR}/**/*", recursive=True)
                 if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))])
    return f"Dataset ready in {CHAR_DIR} with {count} images."

def launch_job(name):
    if name not in SCRIPTS:
        return f"Unknown job: {name}"
    if name in PROCS and PROCS[name].poll() is None:
        return f"{name} already running."
    cmd = f'{SCRIPTS[name]}'
    t = threading.Thread(target=_run_bg, args=(name, cmd), daemon=True)
    t.start()
    return f"Started: {name}"

def stop_job(name):
    p = PROCS.get(name)
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill()
        return f"Stopped: {name}"
    return f"No running job: {name}"

def tail_logs(name, lines):
    path = LOGFILES.get(name)
    if not path or not os.path.exists(path):
        return "(no logs yet)"
    try:
        cmd = f"tail -n {int(lines)} {shlex.quote(path)}"
        out = subprocess.run(["/bin/bash", "-lc", cmd], capture_output=True, text=True)
        return out.stdout or "(empty)"
    except Exception as e:
        return f"(tail error) {e}"

def check_env():
    checks = []
    for p in [
        f"{WORKDIR}/configs/dataset_i2v.json",
        f"{WORKDIR}/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        f"{WORKDIR}/models/vae/wan_2.1_vae.safetensors",
        f"{WORKDIR}/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors",
        f"{WORKDIR}/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors",
    ]:
        checks.append(("✅" if os.path.exists(p) else "❌") + " " + p)
    n = len([p for p in glob.glob(f"{CHAR_DIR}/**/*", recursive=True)
             if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))])
    checks.append(f"📁 {CHAR_DIR} — {n} images")
    return "\n".join(checks)

with gr.Blocks(title="WAN 2.2 I2V LoRA Trainer") as demo:
    gr.Markdown("# WAN 2.2 I2V LoRA Trainer\nSimple control panel for dataset + training")

    with gr.Row():
        url = gr.Textbox(label="Dataset URL (zip or tar.*)", placeholder="https://example.com/dataset.zip")
        dl_btn = gr.Button("Download / Extract")
    dl_out = gr.Textbox(label="Dataset status", lines=3)
    dl_btn.click(download_dataset, inputs=url, outputs=dl_out)

    gr.Markdown("### Environment Check")
    refresh = gr.Button("Refresh")
    env_out = gr.Textbox(label="Status", value=check_env(), lines=8)
    refresh.click(fn=check_env, inputs=None, outputs=env_out)

    gr.Markdown("### Training Controls")
    with gr.Row():
        hn_start = gr.Button("Train HN")
        hn_stop = gr.Button("Stop HN")
        ln_start = gr.Button("Train LN")
        ln_stop = gr.Button("Stop LN")

    hn_log_lines = gr.Slider(50, 2000, value=200, step=10, label="HN log lines")
    hn_log = gr.Textbox(label="HN log tail", lines=20)
    ln_log_lines = gr.Slider(50, 2000, value=200, step=10, label="LN log lines")
    ln_log = gr.Textbox(label="LN log tail", lines=20)

    hn_start.click(lambda: launch_job("Train HN"), outputs=hn_log)
    hn_stop.click(lambda: stop_job("Train HN"), outputs=hn_log)
    ln_start.click(lambda: launch_job("Train LN"), outputs=ln_log)
    ln_stop.click(lambda: stop_job("Train LN"), outputs=ln_log)

    gr.Timer(2.0, fn=lambda l: tail_logs("Train HN", int(l)), inputs=hn_log_lines, outputs=hn_log)
    gr.Timer(2.0, fn=lambda l: tail_logs("Train LN", int(l)), inputs=ln_log_lines, outputs=ln_log)

token = os.getenv("UI_TOKEN", "")
if token:
    demo.launch(server_name="0.0.0.0", server_port=7860, auth=("admin", token))
else:
    demo.launch(server_name="0.0.0.0", server_port=7860)
