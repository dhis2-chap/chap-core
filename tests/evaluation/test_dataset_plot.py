from chap_core.plotting.dataset_plot import StandardizedFeaturePlot


def test_standardized_feautre_plot(simulated_dataset):
    plotter = StandardizedFeaturePlot.from_dataset_model(simulated_dataset)
    plotter.plot()
