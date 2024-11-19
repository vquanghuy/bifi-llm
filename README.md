# Automated Repair Service based on Break-It-Fix-It research and LLM

## Install dependencies

```bash
pip install flax transformers accelerate gdown scikit-learn datasets bitsandbytes peft levenshtein fairseq flask black
apt install unzip
```

## Data

### Dataset

- [github-python-test.zip](https://drive.google.com/file/d/17edjmroalbiDNSX2WY10lG2F8H4OmYuN/view?usp=sharing)
- [prutor-deepfix.zip](https://www.cse.iitk.ac.in/users/karkare/prutor/prutor-deepfix-09-12-2017.zip)

### Trained model

- [BIFI Fixer Round 2](https://drive.google.com/file/d/1ZFdVEZhUkaO70IVxFhDTWfxrXPS5Dw2H/view?usp=drive_link)

Gdown

```
gdown https://drive.google.com/uc?id=17edjmroalbiDNSX2WY10lG2F8H4OmYuN
```

## Usage

1. **Clone the Repository:**

```bash
git clone git@github.com:vquanghuy/bifi-llm.git
cd bifi-llm
```
2. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

## Code Fixer API

Navigate to the server folder and run command

### For debug

```bash
python app.py
```

### For production

```bash
waitress-serve --port=8080 --call app:create
```

## Docker Image

- Image [2.3.1-cuda12.1-cudnn8-devel](https://hub.docker.com/layers/pytorch/pytorch/2.3.1-cuda12.1-cudnn8-devel/images/sha256-a22a1fca37f8361c8a1e859cd6eb6bd9d1fb384f9c0dcb2cfc691a178eb03d17?context=explore)

