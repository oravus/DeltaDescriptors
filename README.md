# Delta Descriptors

Source code for the paper - "Delta Descriptors: Change-Based Place Representation for Robust Visual Localization", published in IEEE Robotics and Automation Letters (RA-L) 2020 and to be presented at IROS 2020.

## Requirements
```
matplotlib==2.0.2
numpy==1.15.2
tqdm==4.29.1
scipy==1.1.0
scikit_learn==0.23.1
```

See `requirements.txt`, generated using `pipreqs==0.4.10` and `python3.5.6`


## Run

#### Describe and Match
Delta Descriptors are defined on top of global image descriptors, for example, NetVLAD. Given such descriptors, compute Delta Descriptors and match across two traverses as below:
``` shell
python src/main.py --genDesc --genMatch --seqLength 16 --descFullPath <full_path_of_desc.npy> --descQueryFullPath <full_path_of_query_desc.npy>
```

The options `--gendesc` and `--genMatch` can be used in isolation or together.

#### Describe only
In order to compute only the descriptors for a single traverse, use:
``` shell
python src/main.py --genDesc --seqLength 16 --descFullPath <full_path_of_desc.npy>
```

#### Match only
For only computing matches, given the descriptors (Delta or some other), use:
``` shell
python src/main.py --genMatch --descFullPath <full_path_of_desc.npy> --descQueryFullPath <full_path_of_query_desc.npy>
```




### Citation
If you find this code or our work useful, cite it as below:
```
@article{garg2020delta,
  title={Delta Descriptors: Change-Based Place Representation for Robust Visual Localization},
  author={Garg, Sourav and Harwood, Ben and Anand, Gaurangi and Milford, Michael},
  journal={IEEE Robotics and Automation Letters},
  year={2020},
  publisher={IEEE},
  volume={5},
  number={4},
  pages={5120-5127},  
}
```

### License
The code is released under MIT License.
