from chap_core.rest_api.v1.routers.visualization import get_available_metrics, generate_visualization, \
    generate_backtest_plots, list_backtest_plot_types


def test_get_available_metrics():
    metrics = get_available_metrics(backtest_id=1)
    print(metrics)
    assert any(metric.id == "detailed_rmse" for metric in metrics)

def _check_generated_visualization(visualization_name, session):
    '''
    Just make the plot and make sure no errors are raised.
    '''


def test_generate_metric_visualization(seeded_session, visualization_name="MetricByHorizonV2Mean", metric_id="detailed_rmse"):
    assert generate_visualization(
        visualization_name=visualization_name,
        backtest_id=1,
        metric_id=metric_id,
        session=seeded_session)


def test_generate_backtest_plot(seeded_session, visualization_name="backtest_plot_1", backtest_id=1):
    assert generate_backtest_plots(visualization_name=visualization_name, backtest_id=backtest_id, session=seeded_session)

def test_all_backtest_plots(seeded_session):
    plot_types = list_backtest_plot_types()
    for plot_type in plot_types:
        plot_name = plot_type.id
        test_generate_backtest_plot(seeded_session, visualization_name=plot_name, backtest_id=1)

