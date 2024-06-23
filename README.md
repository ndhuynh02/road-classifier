This is the Road classifier project

To train the model, please use the `main` function in `src/train.py`
or just call 
```bash
python src/train.py
```
to use default configurations
After training, the model is automatically scripted and saved.

Function `infer` in `src/infer.py` is for inferance. 
Its parameter is 
- `ckpt_path`: path to `.pt` file
- `image_paht`: path to image file
```bash
python src/infer.py
```