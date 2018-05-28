# drl4cellmovement

This is the source code of the paper Deep Reinforcement Learning of Cell Movement in the Early Stage of C. elegans Embryogenesis.

Original paper (accepted by Bioinformatics https://doi.org/10.1093/bioinformatics/bty323): https://arxiv.org/abs/1801.04600.

## Prerequisites

Python3

Mesa

Pytorch

Numpy

matplotlib.pyplot

## Usage

<pre>
python run_dqn.py
</pre>

## Code structure
./nuclei/: contains the C. elegans embryogenesis data, used for initialization, and the movements for the environmental cell

./draw_plan.py: code for the visualization of the dynamic embryo, using tkinter

./model.py: core code of the agent-based modeling environment for cell movement

./run_dqn.py: core code for training the migration cell moving under certain regulatory mechanisms

## Paper citation

If you use this code, please consider citing the following paper at your convenience:

<pre>
@article{wang2018deep,
  title={Deep Reinforcement Learning of Cell Movement in the Early Stage of C. elegans Embryogenesis},
  author={Wang, Zi and Wang, Dali and Li, Chengcheng and Xu, Yichi and Li, Husheng and Bao, Zhirong},
  journal={arXiv preprint arXiv:1801.04600},
  year={2018}
}
</pre>

## Contact

If you have any questions, Please contact the author Zi Wang (zwang84 at vols dot utk dot edu), the University of Tennessee, Knoxville.
