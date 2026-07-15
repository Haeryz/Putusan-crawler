# Run from local VS Code on a RunPod GPU machine

The recommended arrangement is:

~~~text
VS Code user interface: local Windows machine
Repository + Python + terminal + GPUs: RunPod
Connection: VS Code Remote - SSH
~~~

There is no safe project setting that changes every ordinary **python
main.py** command in a local shell into an SSH command. Instead, open the
repository as a remote workspace. VS Code still runs on the local computer,
but every new integrated terminal in that window runs on RunPod automatically.

## 1. Create the Pod

Create a RunPod Pod with:

- two A100 80GB GPUs;
- the RunPod PyTorch template;
- a persistent network volume mounted at **/workspace**;
- a public IP and exposed TCP port 22 for full SSH support.

The volume matters more than it first appears: container storage is erased on
every **Stop**, not only when a Pod is replaced. Keep the repository, the
environment, and the model caches on **/workspace**. See section 7.

## 2. Configure SSH on the local machine

Create an Ed25519 key if needed:

~~~powershell
ssh-keygen -t ed25519
~~~

Add the complete contents of **~/.ssh/id_ed25519.pub**, including the
**ssh-ed25519** prefix, to RunPod account settings. RunPod injects account keys
when the Pod is **created**; a key added afterwards is not in the running Pod's
authorized_keys until it is recreated.

### Load the key into the Windows ssh-agent

Do this before configuring the host. Without an agent, a passphrase-protected
key is re-prompted on every connection, and Remote-SSH opens several per
session. The service ships **Disabled** on Windows, so it must be enabled once
from an **Administrator** PowerShell:

~~~powershell
Set-Service ssh-agent -StartupType Automatic
Start-Service ssh-agent
ssh-add $env:USERPROFILE\.ssh\id_ed25519
~~~

The Windows agent stores the key encrypted in the registry, so the passphrase
survives reboots — `ssh-add` is a one-time cost, not a per-session one. Use the
Windows OpenSSH agent specifically: VS Code drives
`C:\Windows\System32\OpenSSH\ssh.exe`, which does not see a Git Bash agent
started with `eval $(ssh-agent)`.

A denial that reads `Server accepts key: ...` immediately followed by
`Permission denied` is this problem, not an authorization problem — the key was
recognised, but signing needed a passphrase nobody could supply.

### Choose a connection form

The Pod's Connect page offers two, and the choice matters across restarts:

| | Basic SSH (proxy) | SSH over exposed TCP |
|---|---|---|
| Address | `ssh.runpod.io`, fixed | public IP + port |
| Survives Stop/Start | yes | **no** — both change |
| SCP / SFTP | not supported | supported |

Prefer the **proxy**: the login is `<pod-id>-<key-hash>`, and because the Pod ID
survives a Stop, the entry below is written once and never edited. The exposed
TCP address changes on every restart.

~~~sshconfig
Host sinergi-runpod
    HostName ssh.runpod.io
    User 5eq6h6kbg5biu0-64410abb
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 6
~~~

Take the exact `User` from the Connect page's SSH command — it is the Pod ID
plus a hash of the key, not the bare Pod ID. It changes only if the Pod is
recreated or the key changes.

`ServerAliveInterval` keeps the session from dropping during the long silences
between training log lines.

**Fallback.** RunPod documents no SCP/SFTP over the proxy. Remote-SSH installs
its server over an exec channel rather than SCP and is expected to work, but if
it stalls while "installing the VS Code Server", switch to exposed TCP and
update `HostName`/`Port` from the Connect page after each restart:

~~~sshconfig
Host sinergi-runpod
    HostName 157.157.221.29
    User root
    Port 15236
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 6
~~~

Verify either form from a local PowerShell terminal:

~~~powershell
ssh sinergi-runpod
~~~

## 3. Open the remote repository in VS Code

1. Install the Microsoft **Remote - SSH** extension locally.
2. Run **Remote-SSH: Connect to Host...** and choose **sinergi-runpod**.
3. In the new remote window, open a terminal and clone the repository onto the
   persistent volume:

   ~~~bash
   cd /workspace
   git clone YOUR_REPOSITORY_URL Sinergi
   ~~~

   Cloning under **/workspace** is required, not stylistic: everything outside
   it is erased on Stop. See section 7.

4. Open the folder with **File > Open Folder > /workspace/Sinergi**. Do not run
   `code .` here — inside an already-remote window it opens a redundant second
   window.
5. Check the lower-left VS Code status bar says **SSH: sinergi-runpod**.
6. Open **Terminal > New Terminal** and verify:

   ~~~bash
   hostname
   pwd
   nvidia-smi
   ~~~

At this point the editor window is displayed locally, but its files, Python
extensions, debugger, and integrated terminals run on RunPod. Opening a local
Sinergi window and opening a RunPod window are different; always check the
lower-left SSH indicator before starting an expensive job.

## 4. Install the GPU environment

In the remote terminal:

~~~bash
cd /workspace/Sinergi
bash trainer/sft/setup_runpod.sh
source .venv/bin/activate
~~~

The setup follows the pinned versions from the original notebook. Run it once
per persistent environment, not before every training run.

## 5. Authenticate without committing secrets

Set credentials through RunPod Secrets/environment variables or export them
only in the remote shell:

~~~bash
export HF_TOKEN="..."
export WANDB_API_KEY="..."
wandb login
~~~

Do not put either token in Python, Git, **.vscode**, or a committed **.env**.
The default W&B project is **putusan-sft**.

## 6. Run the complete job

Confirm the hardware profile first — see section 8. The shortest command is:

~~~bash
cd /workspace/Sinergi
source .venv/bin/activate
cd trainer/sft
python main.py
~~~

**main.py** sees that it is not under DDP and automatically relaunches itself
with two workers. It then runs these stages:

1. validate the two A100 80GB GPUs and DDP ranks;
2. initialize W&B;
3. load Qwen and attach LoRA;
4. load and format the datasets;
5. measure/select context length;
6. create the response-only trainer;
7. train and evaluate;
8. save the adapter locally;
9. upload the adapter directory as a versioned W&B model artifact;
10. wait for upload completion and finish the W&B run.

Equivalent repository-root command:

~~~bash
python trainer/sft/main.py \
  --max-steps 100 \
  --wandb-project putusan-sft \
  --wandb-artifact-name qwen-extractor-sft-lora
~~~

You can also run **Terminal > Run Task > SFT: Train on connected RunPod**.
That task uses the remote workspace's **.venv**; it does not establish SSH by
itself.

For a long unattended run, use **tmux** so an SSH interruption does not stop it:

~~~bash
tmux new -s putusan-sft
cd /workspace/Sinergi/trainer/sft
source ../../.venv/bin/activate
python main.py
# Detach with Ctrl+B, then D. Reattach with: tmux attach -t putusan-sft
~~~

## 7. Persistence across Stop

**Stop** releases the GPUs, keeps the Pod ID, preserves the volume disk
(**/workspace**), and **erases the container disk** — everything else, including
**/root**. **Terminate** deletes the Pod outright and keeps nothing but a
network volume. Volume storage is billed while stopped.

The container-disk wipe silently defeats any cache under **/root**. Redirect the
caches onto the volume, once, by appending to **~/.bashrc** on the Pod:

~~~bash
export HF_HOME=/workspace/.cache/huggingface
export WANDB_CACHE_DIR=/workspace/.cache/wandb
~~~

**HF_HOME** is the one that pays for itself. Its default is
**~/.cache/huggingface**, on the container disk, so without this the Qwen base
model and the dataset re-download after every single Stop. Nothing reports an
error; the run just starts slowly.

Already safe, because they resolve under the repo on the volume:

- **.venv** — so section 4 is once per Pod, not once per restart
- **outputs/sft/cache** — the cached token-length measurement
- **outputs/sft/checkpoints/**, **qwen_extractor_sft_lora/**

`setup_runpod.sh` installs **uv** into **$HOME/.local/bin**, on the container
disk, so `uv` vanishes on Stop. Harmless: **.venv** persists and activates
without it. Re-running the setup script simply reinstalls `uv` first.

Shell exports vanish too. Keep **HF_TOKEN** and **WANDB_API_KEY** as RunPod
Secrets to avoid re-exporting them each restart.

### Resuming after a Stop

1. Start the Pod, and confirm it actually got its GPUs (section 8).
2. **Remote-SSH: Connect to Host > sinergi-runpod** — no config edit, if you
   used the proxy form.
3. `cd /workspace/Sinergi && git pull`
4. tmux, activate **.venv**, `python main.py`.

## 8. Hardware profile

`validate_hardware` enforces the two-A100 profile and aborts at stage 1 of 11,
before any training work:

- exactly **2** DDP workers — `main.py` relaunches itself with
  `required_gpu_count`, and a different `WORLD_SIZE` is rejected;
- every device name must contain **A100**;
- every device needs at least **78 GiB**.

Check after each restart, because RunPod **may allocate zero GPUs** on restart
if capacity has shifted — a failed restart is a capacity problem, not a bug:

~~~bash
nvidia-smi --query-gpu=name,memory.total --format=csv
~~~

- Two A100 80GB: proceed.
- Right count, wrong model (2x A40, 2x H100): add **--allow-non-a100**, which
  skips the name and VRAM checks but still requires CUDA. An H100 Pod passes the
  count check and fails the name check without it.
- Not exactly two GPUs: **--allow-non-a100 does not help** — the worker count is
  a separate check. Resize the Pod, or change `required_gpu_count` /
  `require_distributed_launch` in **config.py**.

## Outputs

- Local/volume adapter: **qwen_extractor_sft_lora/**
- Trainer checkpoints: **outputs/sft/checkpoints/**
- W&B metrics: project **putusan-sft**
- W&B model artifact: **qwen-extractor-sft-lora:latest**

Training returns success only after the W&B artifact upload completes.

## Official references

- [RunPod SSH connections](https://docs.runpod.io/pods/configuration/use-ssh)
- [RunPod manage Pods: stop vs terminate](https://docs.runpod.io/pods/manage-pods)
- [RunPod storage types](https://docs.runpod.io/pods/storage/types)
- [VS Code Remote - SSH](https://code.visualstudio.com/docs/remote/ssh)
- [Windows OpenSSH key management](https://learn.microsoft.com/windows-server/administration/openssh/openssh_keymanagement)
- [W&B model artifacts](https://docs.wandb.ai/models/artifacts/construct-an-artifact)

