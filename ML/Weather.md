Weather forecasts have become much more accurate: https://ourworldindata.org/weather-forecasts


Nature paper: https://www.nature.com/articles/s41586-023-06185-3
Bi K, Xie L, Zhang H, Chen X, Gu X, Tian Q. Accurate medium-range global weather forecasting with 3D neural networks. 
Nature. 2023 Jul;619(7970):533-538. doi: 10.1038/s41586-023-06185-3. Epub 2023 Jul 5. 
Erratum in: Nature. 2023 Sep;621(7980):E45. PMID: 37407823; PMCID: PMC10356604.


https://github.com/pvigier/perlin-numpy

The code base of Pangu-Weather was established on PyTorch, a Python-based library for deep learning. 
In building and optimizing the backbones, Swin transformer was used: https://github.com/microsoft/Swin-Transformer. 
 
The computation of the CRPS metric relied on the xskillscore package, https://github.com/xarray-contrib/xskillscore/. 

The implementation of Perlin noise was inherited from a GitHub repository, https://github.com/pvigier/perlin-numpy. 

We also used other Python libraries, such as NumPy and Matplotlib, in the research project. We released the trained models, inference code and the pseudocode of details to the public at a GitHub repository: https://github.com/198808xc/Pangu-Weather (https://doi.org/10.5281/zenodo.7678849). The trained models allow the researchers to explore Pangu-Weather’s ability on either ERA5 initial fields or ECMWF initial fields, 
where the latter is more practical as it can be used as an API for almost real-time weather forecasting.

DATA
For training and testing Pangu-Weather, a subset of the ERA5 dataset (around 60 TB) was used:
https://cds.climate.copernicus.eu/, the official website of Copernicus Climate Data (CDS). 


For comparison with operational IFS, he forecast data and tropical cyclone tracking results of ECMWF: 
https://confluence.ecmwf.int/display/TIGGE, the official website of the TIGGE archive. 
Truth routes of tropical cyclones from the International Best Track Archive for Climate Stewardship (IBTrACS) project, 
https://www.ncei.noaa.gov/products/international-best-track-archive. 


Weather raw data: https://weather.cod.edu/forecast/
