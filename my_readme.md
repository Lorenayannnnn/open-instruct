

# Setup env

- Set up from root:
  ```
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda
  ```
  - to ~/.bashrc, add:
    ```
    export PATH="$HOME/miniconda/bin:$PATH"
    ```
  - restart terminal or run:
    ```
    source ~/.bashrc
    ```
  - ```conda init```
  - Verify installation:
    ```
    conda --version
    ```

- conda env:
    ```bash
    conda create -n open-instruct python=3.10
    conda activate open-instruct
    pip install .
    pip install beaker-py==1.32.2
    ```