import warnings
warnings.filterwarnings("ignore")
import os
import torch
import argparse
from PIL import Image
from utils import transforms
from config import cfg
from utils.logger import setup_logger
from process_for_test_CCK import do_inference
from gan.model import Model

torch.multiprocessing.set_sharing_strategy('file_system')


def get_one_img(query_img_path, transform):

    img = Image.open(query_img_path).convert("RGB")
    img = transform(img)
    pid = int(query_img_path[-24:-20])
    camid = int(query_img_path[-18:-15])

    return {'origin': img,
            'pid': pid,
            'camid': camid,
            'trackid': -1,
            'file_name': query_img_path
            }


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    normalizer = transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    transform = transforms.Compose([transforms.RectScale(256, 256),
                                    transforms.ToTensor(),
                                    normalizer])

    query_data = get_one_img('../AIC21/veri_pose/query/0002_c002_00030600_0.jpg', transform=transform)
    model = Model()
    model.reset_model_status()
    do_inference(cfg, model, query_data, None)


if __name__ == '__main__':
    main()
