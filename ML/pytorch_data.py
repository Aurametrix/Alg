from quilt.data.akarve import BSDS300 as bsds
from quilt.asa.pytorch import dataset

return bsds.images.train(
    asa=dataset(
        include=is_image,
        node_parser=node_parser,
        input_transform=input_transform(...),
        target_transform=target_transform(...)
    ))
