

def get_results_path():
    path = Path(__file__).parent.parent.parent / 'results'
    path.mkdir(exist_ok=True)
    return path