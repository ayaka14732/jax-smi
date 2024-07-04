import glob

# https://github.com/yixiaoer/tpux/blob/1c158a568c6164dff93047f8f13b0bf1e8838725/src/tpux/cli.py#L215
ON_TPU = len(glob.glob('/dev/accel*')) > 0
