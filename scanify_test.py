from scanify import Scanify, ScanifyConfig

cfg = ScanifyConfig(
    keep_color_debug=True,
    save_debug=True,  # turn on saving
    save_dir="stage0_debug",  # optional; defaults here if None
    save_every_step=True,  # dump all key steps
    do_binarize=False,  # keep color; skip hard thresholding
)
stage0 = Scanify(cfg)
res = stage0.process("input/2.jpeg")  # saves under pred/stage0_debug/photo/...
