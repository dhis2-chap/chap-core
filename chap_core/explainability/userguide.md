# Explainability module

The explainability module in the chap_core repo is designed to allow users to obtain a quantitative explanation for a specific prediction made by some user model. More specifically, it leverages the LIME algorithm with interchangeable pipeline modules to produce importance weighting to segments of the input model-agnostically.

This markdown file provides a short user guide on how to use the explainability module in chap_core. It begins with some background on what explainability and lime is, before outlining how to call the functions in the module, and what configuration options are available.

## Background

#### Explainability

The chap-core repo provides a singular cohesive ecosystem for running, evaluating and comparing user-defined models, for the prediction of climate-health time series data. Many of the most well-performing models on such data, however, have complex and opaque architectures which give little insight into _why_ it produces the results that it does - so called "black box" models. For instance, deep neural networks are a popular family of architectures that can flexibly adapt to large input datasets to produce very accurate results, predicting future data with high accuracy, but because of the uninterpretable nature of neural networks, it can be very difficult for a human to understand the reasoning behind the specific predictions they make.

Explainability is the area of research in machine learning dedicated to providing explanations for predictions of black box models. This is often done by giving an importance weighting to the input features, where you assign a value to each of the inputs, denoting how much that specific feature contributed positively or negatively to the prediction - i.e., how much the model gave its prediction _because of_ or _despite_ that input value.

#### LIME

LIME (Local Interpretable Model-Agnostic Explanations) is one algorithm which seeks to provide interpretable explanations for predictions of black box models. As the name suggests, two of the defining characteristics of this approach is that it is **local** - meaning that it provides explanations for a single prediction, unlike global explanations, which can explain all possible predictions at the same time - and that it is **model-agnostic**, meaning that it works for any and all models, not just a specific architecture.

LIME works by creating a dataset of "perturbations", i.e. taking the original input vector you want to explain [1], and creating many similar input vectors where you have randomly set a couple of the feature values to zero, effectively "turning them off". You can then feed the perturbed inputs back into the model and get their predictions, and thus have a training dataset of inputs and outputs [2]. With this, you can train another **interpretable** model (called the surrogate model), like linear regression or decision trees, which should then behave similarly to the original black box model in the neighborhood of the original input. You can thereby read the importance weighting from the interpretable model, which should be similar to the weighting of the black box model.

#### Segmentation

The models integrated in chap_core utilizes time series data, which means that the input for a specific prediction can contain several hundreds of values over multiple temporal features. If we were to naïvely perturb ... we would need a huge dataset to sufficiently cover the entire n-dimensional neighborhood of the original input, and obtaining the predictions for each of the perturbations would take way too long - not to mention that the importance weighting for a single value in an entire time series doesn't really provide much information without the context of the neighboring values. You are usually more interested in the impact of a sudden spike, or long term dip in a time series, than the impact of the specific value at a specific time for a specific variable.

What we want, is for LIME to provide us with importance weighting for _segments_ of the input data, such that we can interpret the effect of entire sections of the input on the output. But segmenting time series data into interpretable sections is not trivial, and we thus meet one of the first challenges in adapting the LIME pipeline for time series. Looking at a time series ourselves, it is often obvious where we'd want to cut up the series into segments; one containing the initial spike, the next encompassing the much larger section where the value doesn't change much, et.c. But automatically segmenting a time series is much more difficult - one reasonable method is to divide the series into equally large sections, but that runs the risk of dividing one section of the series which naturally belongs together, across two segments. Another method could be to divide the possible values into high, medium, and low values, and segmenting based on sections containing only low, medium or high values - but that would split up a spike into six segments.

The explainability module provides the user with an assortment of possible algorithms to segment the input time series, and they can themselves choose which to use when calling the LIME pipeline.


#### Perturbation

Another difficulty in using the LIME approach on time series data, is that there isn't an immediate obvious solution on how to "turn off" a segment of the input. When using images as input to a model, one could blur or black out sections of the image to effectively turn that section off for the model, but there is no immediate equivalent to that in time series data. Simply setting all the values in a segment to zero, for instance, is likely not a good choice for making that segment as unimpactful on the final output as possible, since a section of all-zero values for some variable would often be semantically meaningful - for instance, setting the amount of rainfall during some period to zero would imply a dry spell, in contrast with what might be the most common or natural rainfall pattern in the area. What one would like to insert into a segment to "turn it off" would be something which is most representative of the background pattern of the variable.

Multiple strategies have been put forward for optimally perturbing time series data, and this module implements many of these, allowing the user to select the strategy they themselves want when calling the LIME pipeline.

#### Surrogate

As mentioned, the LIME algorithm works by training an explainable model - called the surrogate model - on a perturbed dataset in the neighborhood of the original input, so that it behaves similarly to the original black box model around that original prediction. The most common surrogate model is a linear regression model, but decision trees are also a viable explainable alternative, though the latter only produces absolute weighted values, saying nothing about the direction of impact. Currently, this module only implements linear regression models, in the form of ridge.


#### Distance

When training the surrogate model, you want to weight the training dataset according to how similar the samples are to the original input - the closer the perturbation, the more that perturbation should affect the surrogate model. A common way to do this, which is also the way presented in the original LIME paper, is to simply count up the number of perturbed features in the perturbed input vector, and transforming that according to some kernel. This means that perturbed vectors with more perturbed features are weighted less than those with fewer perturbed features. This also works for time series data, where you count how many segments have been replaced/turned off, but if using a non-uniform segmenter, this doesn't respect the length of the segments - replacing a very long segment would make the entire time series more dissimilar from the original than replacing a very small one. Alternative distance algorithms therefore consider the total dissimilarity between replaced segments, rather than just counting it as either replaced or not replaced.


## Running the LIME pipeline in the explainabiltiy module

#### Prerequisites

[TODO, currently the model is trained within the lime pipeline, in the future the pipeline will take the pickle or something similar as input]

#### Running LIME

The LIME pipeline from the explainability module is available through the command line interface (CLI) using the ```explain-lime``` command. It is run by calling the explain command along with the name of the model to explain, the location of the dataset on which to explain, the location on which to explain, and the number of time steps into the future on which to explain (called the horizon).

As previously mentioned, the LIME algorithm only works for local explanations, i.e. on a particular prediction. For a time series predictor, one particular prediction is defined by the input data for a particular location, for a specific dataset, at a specific time in the future (since the model may predict for several time steps into the future; all considered individual predictions).

There are also currently two main lime pipeline functions; explain() and explain_adaptive(). The former is the standard LIME pipeline, while the latter uses the Expected Active Gain for Local Explanations algorithm (EAGLE) to produce a surrogate dataset adaptively, aiming to optimize perturbation selection in order to reduce the number of calls to the original black box model.

An example of a simple run with the ```explain``` command is:


```bash
chap explain-lime --model_name https://github.com/sandvelab/chap_auto_ewars_weekly@737446a7accf61725d4fe0ffee009a682e7457f6 --dataset_csv example_data/nicaragua_weekly_data.csv --location boaco --horizon 3
```

Additionally, there are multiple arguments with which to customize the LIME pipeline, using the lime-params prefix:

### Options

#### ```granularity``` (Default: 6)
The number of segments to divide the time series input into, if the segmenter does not calculate this automatically.

#### ```segmenter_name``` (Default: uniform)
Name of which segmentation strategy to use. May take any of the following (where granularity is not mentioned, the number of segments is determined automatically):
- **"uniform"**: Segments the time series uniformly, dividing them into ```granularity``` segments.
- **"exponential"**: Segments the time series exponentially, with smaller segments at the beginning of the series, growing exponentially, with the largest segment most recently.
- **"reverse_exponential"**: Segments the time series exponentially, with smaller segments at the end of the series, with exponentially larger segments earlier in the series.
- **"matrix_slope"**: Segments the time series into ```granularity``` segments, by differentiating the matrix profile of the series. A matrix profile is a calculation of how similar a section of a data series is to its most similar section elsewhere in the series. Every data point in the series is assigned a value, and this similarity series is differentiated to find where the similarity changes the quickest in the series. Segment beginnings/endings are added at these points until achieving the desired number of segments.
- **"matrix_diff"**: Segments the time series into ```granularity``` segments, by finding the largest differences in matrix profile similarities between neighboring points in the series. A matrix profile is a calculation of how similar a section of a data series is to its most similar section elsewhere in the series. Every data point in the series is assigned a value, and by finding the data points where its similarity differs the most from one of its neighboring points, we add segment boundaries at these points until achieving the desired number of segments.
- **"matrix_bins"**: Segments the time series by finding contiguous sections of equally binned similarty values from matrix profiling. A matrix profile is a calculation of how similar a section of a data series is to its most similar section elsewhere in the series. Every data point in the series is assigned a value, and the similarity value is horizontally binned - i.e. divided into highest, second highest, ..., n highest category of value, where n is typically a low number. Sections of contiguous points whose similarity value is in the same bin become a segment.
- **"sax"**: Segments the time series by finding contiguous sections of equally binned values using a sax transformation. A sax transformation converts a data series into "words" by horizontally binning the series - i.e. dividing the values into highest, second highest, ..., n highest category of value, where n is typically a low number, and assigning each category a letter. Segments are constructed by finding contiguous runs of values in the same category/letter. The number of horizontal bins is increased until the number of segments roughly equals ```granularity```.
- **"nn"**: Segments the time series into ```granularity``` segments by finding breaks in the correlation of similarity from matrix profiling to the most similar other section of the series. Using matrix profiling, the algorithm identifies, for every point, the index of the time series which is most similar to said point, apart from the point itself. If a section of a time series behaves regularly, then the neighbors of a data point should have most similar indexes which are also neighbors of the most similar index to the original point - i.e., when progressing one step along the time series, the correlated similar index should also move by just one step. If it doesn't, it is considered a break. The algorithm sorts the severity of the break by a heuristic using the mean and variance of the two sides, and selects the desired number of segments from them.


#### ```sampler_name``` (Default: background)
Name of the perturbation strategy to use. In all cases, the perturbed segment has the same length after replacement as before replacement. May take any of the following:
- **"background"**: Replaces the segment with a random section from elsewhere for that variable in the entire dataset (across all locations).
- **"linear"**: Replaces the segment with a linear series from the last value of the previous segment to the first value of the next segment - i.e. connecting the two segments on either side with a straight line.
- **"constant"**: Replaces the segment with a constant series of all zeros.
- **"local_mean"**: Replaces the segment with a constant series of the local mean, i.e. the mean of the values in the replaced segment.
- **"global_mean"**: Replaces the segment with a constant series of the global mean, i.e. the mean of all values in the entire variable series for that location.
- **"random"**: Replaces the segments with a series of random values, selected uniformly between the lowest and highest value in the entire variable series across all locations.
- **"fourier"**: Replaces the segment with a segment from the most prominent frequency of the time series, as calculated from the short time fourier transform of the entire variable series.


#### ```surrogate_name``` (Default: ridge)
Name of the surrogate model to use. Currently only has one implementation:
- **"ridge"**: A ridge regression model, where the importance weighting is calculated from the weights of the model


#### ```weighter_name``` (Default: pairwise)
Name of the perturbation weighting strategy to use. May take any of the following:
- **"pairwise"**: Weights the perturbations by counting the number of perturbed segments, transformed using an RBF kernel.
- **"dtw"**: Weights the perturbations by finding the dynamic time warping (dtw) similarity between the original and perturbed series, transformed using an RBF kernel.


#### ```num_perturbations``` (Default: 300)
Number of perturbations to create for the training of the surrogate model. A higher number will result in a longer running time, as each perturbation must be run through the model for an output, but will also result in a more accurate explanation.

#### ```timed``` (Default: False)
Flag for whether to print timing debug logs during execution.

#### ```adaptive``` (Default: False)
Flag for whether to run the LIME pipeline adaptively or non-adaptively.


An example run using all options is:

```bash
chap explain-lime   --model_name https://github.com/sandvelab/chap_auto_ewars_weekly@737446a7accf61725d4fe0ffee009a682e7457f6   --dataset_csv example_data/nicaragua_weekly_data.csv   --location boaco   --horizon 3   --lime-params.surrogate-name ridge   --lime-params.segmenter-name uniform   --lime-params.sampler-name fourier   --lime-params.granularity 8   --lime-params.num-perturbations 50   --lime-params.timed   --lime-params.adaptive
```



### Output

Running the ```explain-lime``` command will produce a printed output as well as a plot. 

The plot shows the original input for each variable in blue, with predicted future values in orange, and with vertical stipled lines delineating the segments. The background colour of each segment shows the importance weighting for that particular segment - a greener colour means that segment contributed positively to the prediction; a redder colour means that segment contributed negatively to the prediction, and a more yellow colour means that segment didn't make too great of a difference one way or the other on the prediction compared to other segments.
The printed output shows the same result in more detail. Features are labeled with the variable names, plus either "_lag_n" for the segment n steps into the past (read from the right), while "_fut_n" means the value (_not_ the segment, as future predicted values are not segmented) n steps into the future (read from the left). The number next to the feature is the importance weighting, on which the features are sorted in absolute values.

TODO: [Plotting is not finished, importance for static and future values is not shown]

<br><br><br><br><br>



---
[1] Often you don't want to create explanations for every single input - for instance, if the input is an image, getting an importance weighting for every single input pixel would still be really difficult to interpret. What you want is an importance weighting for _interpretable_ parts of the image; whole areas which fit together meaningfully, like a person or an object in the image. What you perturb in such a case is then not the entire vector of inputs, but an "interpretable vector" a smaller vector of the combined interpretable features in the input. The principle from there is the same as described above.

[2] Intuitively, looking at the difference between two outputs when some feature is present vs. not present should tell you something about how important that feature is to the output - this is what the dataset and surrogate model is trying to encompass.