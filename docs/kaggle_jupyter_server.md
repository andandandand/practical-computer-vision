
# Setting up Access to Kaggle Jupyter server in VS Code

You can use Kaggle as a remote backend for VS Code. This lets you work with local coding agents while using Kaggle GPUs for compute.

**Prerequisite:** Install VS Code with the Python and Jupyter extensions.

1. Create a Kaggle notebook by opening [Kaggle Code](https://www.kaggle.com/code) in your browser.

2. Enable GPU acceleration by selecting `Settings → Accelerator → GPU T4 x2`.

   If you cannot enable a GPU yet, follow the instructions in the [Kaggle GPU and TPU guide](https://github.com/andandandand/practical-computer-vision/blob/main/docs/kaggle-gpu-tpu-guide.md).

3. Start the session by selecting `Run → Start Session`. You can also press `Shift + Enter` while a code cell is selected.

4. Select `Run → Kaggle Jupyter Server` to obtain the server URL.

5. Copy the **VS Code Compatible URL**. It will look similar to this:

   ```text
   https://kkb-production.jupyter-proxy.kaggle.net/k/3314.../proxy
   ```

6. In VS Code or Cursor, open or create a Jupyter notebook. Open the Command Palette by pressing `Cmd + Shift + P` on macOS or by selecting **Show and Run Commands** from the search bar.

7. Complete the following steps:

   1. Select `Notebook: Select Notebook Kernel`.
   2. Select `Existing Jupyter Server`.
   3. Paste the URL you copied from Kaggle, then press `Enter`.
   4. VS Code will display a shortened version of the URL. Press `Enter` to confirm it.
   5. Select `Python 3 (ipykernel)`.

8. Create a cell in the VS Code notebook and run:

   ```python
   !nvidia-smi
   ```

   You should see output similar to the following:

   ```text
   Tue Jun 30 06:38:42 2026
   +-----------------------------------------------------------------------------------------+
   | NVIDIA-SMI 580.159.04             Driver Version: 580.159.04     CUDA Version: 13.0     |
   +-----------------------------------------+------------------------+----------------------+
   | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
   |                                         |                        |               MIG M. |
   |=========================================+========================+======================|
   |   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
   | N/A   35C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
   |                                         |                        |                  N/A |
   +-----------------------------------------+------------------------+----------------------+
   |   1  Tesla T4                       Off |   00000000:00:05.0 Off |                    0 |
   | N/A   32C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
   |                                         |                        |                  N/A |
   +-----------------------------------------+------------------------+----------------------+

   +-----------------------------------------------------------------------------------------+
   | Processes:                                                                              |
   |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
   |        ID   ID                                                               Usage      |
   |=========================================================================================|
   |  No running processes found                                                             |
   +-----------------------------------------------------------------------------------------+
   ```

This notebook will have access to all datasets that are accessible on the Kaggle notebook and will save into its filesystem, not your local drive. When evaluating `!pwd` on a code cell, you will get `/kaggle/working` as output. 
