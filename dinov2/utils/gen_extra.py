"""python -m dinov2.utils.gen_extra"""

def gen_extra(data_name: str, root: str, extra: str=None, webdataset: bool=False) -> None:
    match data_name:
        # ImageNet-1k
        case "ImageNet":
            from dinov2.data.datasets import ImageNet
            for split in ImageNet.Split:
                dataset = ImageNet(split=split, root=root, extra=extra if extra else root, webdataset=webdataset)
                dataset.dump_extra()


# python -m dinov2.utils.gen_extra
if __name__ == "__main__":
    # Change the following variables as needed
    data_name = "ImageNet"
    root = r"/root/autodl-tmp/imagenet"
    extra_root = root

    gen_extra(data_name=data_name, root=root, extra=extra_root, webdataset=False)