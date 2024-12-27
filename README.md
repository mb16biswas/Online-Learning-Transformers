# Online-Learning-Transformers

Find the preprint [here](https://arxiv.org/pdf/2409.10242).

![Overall Pipeline](https://github.com/user-attachments/assets/d2c18642-afad-4b22-8f03-2d67b38a48b9)



## Instructions to run the code

### Required directory structure to save the results:

```

|   +-- Base/
|   |   +--Results/
|   |   |  +--res/
|   |   |  +--loss-curves//
|   +-- Data/
|   |   +--data1
|   |   +--data2
|   |   +--data3
|   |   ...

...
```
### Install the required packages:

```
pip install -r requirements.txt

```

### Run the command for  training

```
python main.py
```

Available arguments:
- `--t`: Dataset Type
- `--n`: Dataset Name
