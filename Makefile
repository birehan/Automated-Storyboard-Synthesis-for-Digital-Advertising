create_env:
    conda create --name myenv python=3.8

install:
    conda activate myenv && pip install -r requirements.txt

run
    conda activate myenv && jupyter notebook notebooks/exploratory_data_analysis.ipynb

test:
    conda activate myenv && python test.py
