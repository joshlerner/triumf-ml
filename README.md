# Real-time Pion Reconstruction using GarNet on `hls4ml`
## Introduction to GarNet
GarNet is a graph neural network designed to fit on an FPGA for fast and accurate HEP tasks. \
\
Firmware systems, unlike GPUs, are unable to efficiently run deep neural networks, but the typical shallow neural network is not robust enough to handle complex reconstruction tasks. The GarNet algorithm balances complexity and size while taking advantage of a few important qualities of the data it is expected to process. \
\
The input data, which we will call $F_{in}$, is a series of clusters, each constaining up to $V_{max}$ cells with four features - $\eta_{cell}$, $\phi_{cell}$, $s_{cell}$, and $E_{cell}$ - the cell postition in eta-phi-sampling space and the cell energy. Already we see a major advantage of using point-cloud representations of topo-clusters. While a topo-cluster image may contain thousands of cells, many of which with negligible energy, the point-cloud only considers the $V$ highest energy cells ($V \leq V_{max}$). The first step of the algorithm is to encode the input data into $i$ learned features per vertex (cell), $f_v^i$, and distances, $d_{av}$ between each vertex (cell) and each aggregator. The internal representation is learned using a single linear activation! \
\
The next step is to gather the learned features $f_v^i$ at the aggregators according to the distance, $d_{av}$
$$h_a^i = \dfrac{1}{V_{max}}\sum_v^Ve^{-d^2_{av}}f_v^i$$
Then, the message is passed back to each vertex. In the simplest case,
$$\tilde{f}^i_{av} = h_a^ie^{-d^2_{av}}$$
The output is decoded again using a single linear activation. \
\
The implementation of GarNet with keras found in `util.Layers.py` and `util.Models.py` is modified from Jan Kieseler's [repository](https://github.com/jkiesele/caloGraphNN.git)
## Using the Tutorial Notebooks
Training the GarNet model requires large amounts of data, which consume a lot of memory when waiting to be processed. The solution is to preprocess data and queue it to be accessed in parallel as the model trains. The implementation found in `util.Generators.py` is modified from Jessica Bohm's [repository](https://github.com/jessicabohm/gn4pions_eastbay/blob/master/gn4pions/modules/data.py). \
\
Everything you need to generate data, train the GarNet model, and convert to HLS are in the tutorial files. \
\
Precision analysis must be run from the CLI with the help of `BuildHLS.py` and the shell scripts and python files in `precisionAnalysis`. \
To scan the performance across different precisions for multiple models, run the following from the `projects` directory: \
`~ bash scanPerformance.sh <name> <gpu>` \
To scan the resource use for various resources, run the following from the `projects` directory: \
`~ bash scanResource.sh <name> <resource> <gpu>` \
The `<name>` parameter is one of 'defaultFraction', 'defaultInteger', 'denseFraction', or 'denseInteger'. The `<resource>` parameter is one of 'Latency', 'DSP', 'LUT', or 'FF'. Since resource scans can take a long time, there is an option to use the previously built models' reported resource usage. The output of these scans are an updated dictionary of values and plots of the scans, stored in a data directory.















