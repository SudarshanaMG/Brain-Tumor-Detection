## Brain Tumor Detection

### Local Setup

- `git clone https://github.com/SudarshanaMG/Brain-Tumor-Detection`

- Create Virtual Environment 
    `pip install virtualenv`
    `virtualenv .venv` 
    or  
    - for Linux : `python3 -m venv .venv`
    - for windows : `py -m venv .venv`

- Activate Virtual Environment 
    - Windows - `.venv\Scripts\activate`
    - Linux - `source venv/bin/activate`

### Dependencies required

- Flask
- tensorflow
- numpy
- opencv-python
- matplotlib
- pandas

### For Starting the server

- Start Server `python app.py`

### Adding trained models

- Save trained models in ".h5" extension (ex: "cnn.h5")
- Create a folder named "models" inside project root directory
- Add the models to this folder 

