try:
    import omnipy
    from omnipy.modules.json.typedefs import JsonScalar
    from omnipy import StrDataset
    from omnipy.api.protocols.public.hub import IsRuntime
    from omnipy.hub.runtime import Runtime
    from omnipy import (
        BytesDataset,
        TableWithColNamesDataset,
        TableWithColNamesModel,
        TaskTemplate,
    )
    from omnipy import Dataset, Model
    from omnipy.data.dataset import MultiModelDataset

except ImportError:
    Dataset = None
    Model = None
    MultiModelDataset = None
    omnipy = None
    JsonScalar = None
    StrDataset = None
    IsRuntime = None
    Runtime = None
    BytesDataset = None
    StrDataset = None
    TableWithColNamesDataset = None
    TableWithColNamesModel = None
    TaskTemplate = None
