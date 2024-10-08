# Cell registration across multiple sessions in large-scale calcium imaging data
This package is an implementation of a probabilistic approach for tracking the same neurons (cell registration) across multiple sessions 
in Ca2+ imaging data. The package includes a GUI that supports the entire registration procedure. 

For more information contact lironsheintuch@gmail.com or join our [slack channel](https://cellreg.slack.com).

## Adapted for use on Linux based server
In general CellReg is left the same, however this adaptation is optimized to run without GUI on our Linux based server. 
In our implementation Cellreg runs via a single matlab script (see Leon_CellReg.m) which is called from within a python jupyter botebook.
For performance optimization and asthetics loading bars (which are not correctly rendered in Jupyter output) have been removed.

Leon Kremers (2024)

## Setting up the repository
We encourage the use of official versions (e.g., v1.5.9) for easier debugging processes. Switch to the releases tab on GitHub and checkout the latest version.

1. Cloning:
`git clone https://github.com/zivlab/CellReg.git`
2. Checkout version:
`git checkout v<major>.<minor>.<bugfix> (e.g., v1.5.9)`
3. Run `CellReg_setup.m`

*CellReg version v1.5.5 includes a fix to an important issue with FOV alignment that was introduced into CellReg in version update v1.5.0 on March 2022.*
*This issue may have resulted in some cases in cells being cut off or disappearing from specific sessions, especially if there are relatively large translations/rotations across sessions in the data.*
*The issue is relevant for everyone who has used any version between v1.5.0-v1.5.3.* 
*In cases where versions between v1.5.0-v1.5.3 were used, it is recommended to switch to a new version (v1.5.5 and onward) and to re-register the data and verify that the same results are obtained.*

## Usage and documentation
To run the cell registration procedure you can either use the full GUI version or access the API directly.
To use the GUI run *GUI\CellReg.m*.
An example of how to use the API is provided in the file *CellReg\demo.m*.


The inputs for the cell registration method are the matrices of spatial footprints of cellular activity (ROIs) of the cells that were detected separately in the different sessions. 
The matrix of the spatial footprints of each session should be of size NxMxK, where N is the number of neurons, M is the number of pixels in the y axis and K is the number of pixels in the x axis.

The main output for the cell registration method is the obtained mapping of cell identity across all registered sessions.
Other outputs include the cell scores, the probability for each cell-pair to the be the same cell, the spatial footprints of the cells after the alignment step, and a log file with all the relevant information regarding the data, registration
configurations, and a summary of the registration results and quality. In addition, important figures are saved automatically in a figures directory. 


An example data set and cell registration results are provided in the *SampleData* directory.


For more information refer to the user manual found in the *Docs* directory.

## Main stages of the cell registration procedure

1. Loading the spatial footprints of cellular activity from the different sessions.

2. Aligning all sessions to a reference coordinate system using rigid-body transformation.

3. Computing a probabilistic model of the spatial footprints similarities
of neighboring cell-pairs from different sessions using the centroid
distances and spatial correlations.

4. Obtaining an initial cell registration according to an optimized registration threshold.

5. Obtaining the final cell registration based on a clustering algorithm.

## References
Sheintuch, L., Rubin, A., Brande-Eilat, N., Geva, N., Sadeh, N., Pinchasof, O., Ziv, Y. (2017). Tracking the Same Neurons across Multiple Days in Ca2+ Imaging Data. *Cell Reports*, 21(4), pp. 1102–1115. doi: 10.1016/j.celrep.2017.10.013.
