# compress-al

Compresses any image to decrease its size using Machine Learning! This is done by intelligenlty lowering the number of colors in an image while keeping it extremely truthful to the original.

Choose how many colors you want, and let compress-al do the magic!

## How-to

1. Upload your image.
2. Choose how many colors you want.
3. Compress!

## Setup

#### 1. Download the repositroy to your sever
```
git clone https://github.com/minim-al/compress-al.git
cd compress-al
```

#### 2. Install the packages listed inside `requirements.txt` (Using a virtual environment is recommended)
- With a virtual environment using `pipenv`
```
pipenv shell
pipenv install -r requirements.txt
```
- Without a virtual environment using `pip`
```
pip install -r requirements.txt
```

#### 3. Run the web application
```
streamlit run Intro.py
```
