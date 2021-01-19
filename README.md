# SealExtraction
Extracting the motive out of a wax stamp seal. Creates shapes out of contours and scores them according to
  - their size (~65% of wax size and ~99% most likely)
  - how close to the middle they are
  - how even their rotations are 
  - symmetry of underlying thresholded image
  - density (meaning relation between pixels of the underlying thresholded image to full shape size)

Part of the Coding Da Vinci Niedersachsen 2020 Hackathon, we scan images of seals and produce 3D object out of them.

## Instruction:
You need Python version >= 3.8

### Installation
```
pip install -e . -r requirements.txt
```

## Usage

For development, after clone, run:
```
pip install -e . -r requirements.txt
```

Options are:
```
usage: sealExtraction.py [-h] -o OUTPUT INPUT

positional arguments:
  INPUT                 input file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output file
```

You may need to add 'python3' beforehand depending on your machine's settings. Aka `python3 sealExtraction.py ...`.

## Example 
![alt text](https://github.com/Siegler-von-Catan/sealExtraction/blob/master/exampleResult/input.jpg)
![alt text](https://github.com/Siegler-von-Catan/sealExtraction/blob/master/exampleResult/output.png)
